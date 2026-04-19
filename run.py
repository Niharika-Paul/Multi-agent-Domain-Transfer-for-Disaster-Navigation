import os
from dotenv import load_dotenv
from google import genai
from hybrid_spatial_index import HybridTokyoGraph

# Load API key
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-2.5-flash"

# ------------------------
# LOAD DATASET
# ------------------------
graph = HybridTokyoGraph("tokyo_full_graph_updated.json")
graph.build_index()   # builds R-tree + KD-tree + snaps all facilities

# Disaster epicentre — change these to move the disaster zone
DISASTER_LAT  = 35.68
DISASTER_LON  = 139.76
DISASTER_TYPE = "earthquake"   # or "flood"
SEVERITY      = 0.3            # 0.0 (minor) → 1.0 (catastrophic)
RADIUS_M      = 2000           # affected radius in metres
AGENT_RADIUS_M = 800           # each agent's local observation radius

# Apply disaster to the graph (damages nodes + disables edges)
disaster_summary = graph.apply_disaster(
    disaster_type=DISASTER_TYPE,
    centre=(DISASTER_LAT, DISASTER_LON),
    radius_m=RADIUS_M,
    severity=SEVERITY,
)

# ------------------------
# BUILD LIVE WORLD STATE
# ------------------------
def build_world():
    """Pull live data from the graph to replace the hardcoded world dict."""
    snap = graph.snapshot_local(DISASTER_LAT, DISASTER_LON, AGENT_RADIUS_M)

    nearest_hospital   = graph.nearest_facility("hospital",     DISASTER_LAT, DISASTER_LON)
    nearest_firestation = graph.nearest_facility("fire_station", DISASTER_LAT, DISASTER_LON)

    # Describe local road connectivity (sample up to 5 road nodes)
    road_sample = snap["road_nodes"][:5]
    road_lines = []
    for node in road_sample:
        neighbours = graph._adj.get(node["id"], [])
        passable = [nid for nid, _ in neighbours
                    if graph.is_edge_passable(node["id"], nid)]
        blocked  = [nid for nid, _ in neighbours
                    if not graph.is_edge_passable(node["id"], nid)]
        road_lines.append(
            f"  Node {node['id']} ({node['lat']:.4f},{node['lon']:.4f}): "
            f"{len(passable)} passable roads, {len(blocked)} blocked roads"
        )
    road_desc = "\n".join(road_lines) if road_lines else "  No road nodes in local area."

    # Describe buildings with population
    building_lines = []
    for b in sorted(snap["buildings"], key=lambda x: -x.get("population", 0))[:5]:
        dmg = graph.get_node_damage(b["id"])
        building_lines.append(
            f"  Building {b['id']} ({b['lat']:.4f},{b['lon']:.4f}): "
            f"pop={b.get('population', 0)}, damage={dmg:.0%}"
        )
    building_desc = "\n".join(building_lines) if building_lines else "  No buildings found."

    map_text = f"""
DISASTER ZONE ({DISASTER_TYPE.upper()}, severity={SEVERITY:.0%}, radius={RADIUS_M}m):
  Epicentre: {DISASTER_LAT}, {DISASTER_LON}
  Nodes affected: {disaster_summary['nodes_affected']}
  Edges (roads) disabled: {disaster_summary['edges_disabled']}

LOCAL ROAD NETWORK (within {AGENT_RADIUS_M}m of epicentre):
{road_desc}

BUILDINGS WITH POPULATION:
{building_desc}
  Total local population at risk: {snap['population']:,}
"""

    status_text = f"""
Disaster type     : {DISASTER_TYPE}
Severity          : {SEVERITY:.0%}
Population at risk: {disaster_summary['population_at_risk']:,}
Hospitals affected: {disaster_summary['hospitals_affected']}
Fire stations affected: {disaster_summary['fire_stations_affected']}

Nearest hospital   : id={nearest_hospital['id']}, lat={nearest_hospital['lat']:.4f}, lon={nearest_hospital['lon']:.4f}
Nearest fire station: id={nearest_firestation['id']}, lat={nearest_firestation['lat']:.4f}, lon={nearest_firestation['lon']:.4f}
"""

    # Route from closest fire station / hospital OUTSIDE the disaster zone
    # (roads at the epicentre may be fully blocked at high severity)
    import math
    def _deg_dist(a, b):
        return math.sqrt((a["lat"] - b["lat"]) ** 2 + (a["lon"] - b["lon"]) ** 2)

    centre_node = {"lat": DISASTER_LAT, "lon": DISASTER_LON}
    min_deg = RADIUS_M / 111_000
    outside_fs   = [f for f in graph.get_fire_stations() if _deg_dist(f, centre_node) > min_deg]
    outside_hosp = [h for h in graph.get_hospitals()     if _deg_dist(h, centre_node) > min_deg]

    path, src_fs, dst_hosp = None, nearest_hospital, nearest_firestation
    if outside_fs and outside_hosp:
        src_fs   = min(outside_fs,   key=lambda n: _deg_dist(n, centre_node))
        dst_hosp = min(outside_hosp, key=lambda n: _deg_dist(n, centre_node))
        path = graph.shortest_path_astar(src_fs["id"], dst_hosp["id"], avoid_damaged=True)

    if path:
        dist_km = graph.path_distance_m(path) / 1000
        tt_min  = graph.path_travel_time_s(path) / 60
        route_hint = (f"Computed safe route (A*): {len(path)} road nodes, {dist_km:.2f} km, "
                      f"~{tt_min:.1f} min travel time "
                      f"(fire station {src_fs['id']} to hospital {dst_hosp['id']}, "
                      f"avoiding damaged roads)")
    else:
        staging = graph.find_staging_zones((DISASTER_LAT, DISASTER_LON), RADIUS_M)
        if staging:
            zone_str = "; ".join(
                f"fire station {z['id']} at {z['dist_to_epicentre_m']}m perimeter"
                for z in staging
            )
            route_hint = (f"No ground route found. Recommended staging zones: {zone_str}. "
                          f"Deploy aerial rescue from these positions.")
        else:
            route_hint = "No safe route found — roads into the disaster zone may be fully blocked."

    task_text = f"""
Coordinate disaster response for {DISASTER_TYPE} at ({DISASTER_LAT}, {DISASTER_LON}).
Evacuate at-risk population ({snap["population"]:,} people nearby) to safety.
Direct rescue units from nearest outside fire station id={src_fs["id"]} to hospital id={dst_hosp["id"]}.
{route_hint}
"""

    return {
        "map":    map_text,
        "status": status_text,
        "task":   task_text,
    }


world = build_world()

shared_memory = {
    "mapping":  "",
    "risk":     "",
    "resource": "",
    "routing":  ""
}

# ------------------------
# MODEL CALL
# ------------------------
def call_model(prompt):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text.strip()

# ------------------------
# AGENT PROMPTS
# ------------------------

MAPPING_PROMPT = """
You are the Mapping Agent for a disaster response simulation in Tokyo.
Analyse the road network data and identify which connections are still reachable.
Focus on passable routes between key infrastructure (hospitals, fire stations).

Output format:
REACHABLE_PATHS:
NodeA -> NodeB (reason)
"""

RISK_PROMPT = """
You are the Risk Agent for a disaster response simulation in Tokyo.
Based on the disaster data, identify the highest-risk zones and populations.
Flag any hospitals or fire stations that are compromised.

Output format:
RISKS:
Location/Node: risk description
"""

RESOURCE_PROMPT = """
You are the Resource Agent for a disaster response simulation in Tokyo.
Assess available hospitals and fire stations. Suggest resource allocation
and flag any capacity or access issues.

Output format:
RESOURCE_UPDATE:
facility: recommendation
"""

ROUTING_PROMPT = """
You are the Routing Agent for a disaster response simulation in Tokyo.
Using the mapping and risk information, determine the safest evacuation
and rescue routes. Prefer routes that avoid damaged nodes and blocked roads.

Output format:
ROUTE:
NodeA -> NodeB -> NodeC (distance, reason)

If no safe path exists:
ROUTE: NO SAFE PATH
"""

# ------------------------
# AGENT RUNNER
# ------------------------
def run_agent(role_prompt):
    input_text = f"""
{role_prompt}

CITY MAP & ROAD NETWORK:
{world['map']}

CURRENT CONDITIONS:
{world['status']}

TASK:
{world['task']}

MESSAGES FROM OTHER AGENTS:
Mapping  : {shared_memory['mapping']  or '(none yet)'}
Risk     : {shared_memory['risk']     or '(none yet)'}
Resource : {shared_memory['resource'] or '(none yet)'}
Routing  : {shared_memory['routing']  or '(none yet)'}
"""
    return call_model(input_text)

from evaluator import SimulationEvaluator
evaluator = SimulationEvaluator()

# ------------------------
# MAIN LOOP
# ------------------------
for step in range(3):
    print(f"\n--- STEP {step+1} ---")

    evaluator.start_step()

    shared_memory["mapping"] = run_agent(MAPPING_PROMPT)
    print("\nMapping:\n", shared_memory["mapping"])

    shared_memory["risk"] = run_agent(RISK_PROMPT)
    print("\nRisk:\n", shared_memory["risk"])

    shared_memory["resource"] = run_agent(RESOURCE_PROMPT)
    print("\nResource:\n", shared_memory["resource"])

    shared_memory["routing"] = run_agent(ROUTING_PROMPT)
    print("\nRouting:\n", shared_memory["routing"])

    evaluator.evaluate_step(step + 1, shared_memory, world)

    if "NO SAFE PATH" in shared_memory["routing"] or "->" in shared_memory["routing"]:
        print("\nSimulation finished.")
        break

evaluator.print_report()
