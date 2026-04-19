"""
run.py — Disaster-Aware Multi-Agent Simulation
-----------------------------------------------
Usage:
    python run.py
    python run.py --lat 35.68 --lon 139.76 --type earthquake --severity 0.5 --radius 3000
    python run.py --episodes 3 --no-map
"""

import os
import sys
import math
import argparse
from dotenv import load_dotenv
from google import genai
from hybrid_spatial_index import HybridTokyoGraph
from evaluator import DisasterEvaluator
from visualize import generate_map

# -----------------------------------------------------------------------
# CLI configuration
# -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Disaster-aware multi-agent simulation")
    p.add_argument("--lat",      type=float, default=35.68,        help="Epicentre latitude")
    p.add_argument("--lon",      type=float, default=139.76,       help="Epicentre longitude")
    p.add_argument("--type",     type=str,   default="earthquake",  help="earthquake|flood")
    p.add_argument("--severity", type=float, default=0.5,          help="Severity [0.0-1.0]")
    p.add_argument("--radius",   type=float, default=2000,         help="Disaster radius in metres")
    p.add_argument("--episodes", type=int,   default=1,            help="Number of episodes")
    p.add_argument("--no-map",   action="store_true",              help="Skip map generation")
    args = p.parse_args()

    errors = []
    if not -90  <= args.lat      <= 90:   errors.append(f"Latitude {args.lat} out of range [-90,90]")
    if not -180 <= args.lon      <= 180:  errors.append(f"Longitude {args.lon} out of range [-180,180]")
    if not 0.0  <= args.severity <= 1.0:  errors.append(f"Severity {args.severity} must be in [0.0,1.0]")
    if args.radius <= 0:                  errors.append(f"Radius must be > 0")
    if errors:
        for e in errors: print(f"[ERROR] {e}")
        sys.exit(1)
    return args


# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "gemini-2.5-flash"

def call_model(prompt: str) -> str:
    return client.models.generate_content(model=MODEL_NAME, contents=prompt).text.strip()


# -----------------------------------------------------------------------
# Agent prompts — strictly structured, concise
# -----------------------------------------------------------------------

MAPPING_PROMPT = """You are the Mapping Agent in a Tokyo disaster response simulation.
Analyse the road network. Identify the top 3 still-passable routes between key infrastructure.

Reply in EXACTLY this format — bullet points only, max 5 lines, no prose:
REACHABLE_PATHS:
- NodeA -> NodeB (road type, why passable)
- NodeC -> NodeD (road type, why passable)
- NodeE -> NodeF (road type, why passable)
"""

RISK_PROMPT = """You are the Risk Agent in a Tokyo disaster response simulation.
Identify the top 3 highest-risk locations and flag compromised facilities.

Reply in EXACTLY this format — bullet points only, max 6 lines, no prose:
RISKS:
- Building <ID>: pop=N, damage=D%
- Building <ID>: pop=N, damage=D%
- Epicentre zone: severity S%, N people trapped
COMPROMISED:
- <Hospital/FireStation ID>: reason
"""

RESOURCE_PROMPT = """You are the Resource Agent in a Tokyo disaster response simulation.
Based on the risk report, assign resources to facilities.

Reply in EXACTLY this format — bullet points only, max 5 lines, no prose:
RESOURCE_UPDATE:
- FireStation <ID> -> Hospital <ID>: deploy rescue unit
- Hospital <ID>: establish as primary triage centre
- Staging zone: position for aerial assets if roads blocked
"""

ROUTING_PROMPT = """You are the Routing Agent in a Tokyo disaster response simulation.
Using mapping + risk data, output the top evacuation and rescue routes.

Reply in EXACTLY this format — bullet points only, max 5 lines, no prose:
ROUTE:
- FireStation <ID> -> Hospital <ID> (X km, Y min)
- Building <ID> -> Hospital <ID> (X km, Y min)
- Staging fallback: FireStation <ID> (aerial)

If no ground route exists write exactly:
ROUTE: NO SAFE PATH
- Staging: FireStation <ID> at Xm from epicentre
"""


# -----------------------------------------------------------------------
# World state
# -----------------------------------------------------------------------

def build_world(graph, cfg: dict, disaster_summary: dict) -> dict:
    lat, lon = cfg["lat"], cfg["lon"]
    snap = graph.snapshot_local(lat, lon, cfg["agent_radius"])

    nearest_h  = graph.nearest_facility("hospital",     lat, lon)
    nearest_fs = graph.nearest_facility("fire_station", lat, lon)

    road_lines = []
    for node in snap["road_nodes"][:5]:
        nb = graph._adj.get(node["id"], [])
        passable = sum(1 for nid, _ in nb if graph.is_edge_passable(node["id"], nid))
        road_lines.append(
            f"  Node {node['id']} ({node['lat']:.4f},{node['lon']:.4f}): "
            f"{passable} passable, {len(nb)-passable} blocked"
        )

    building_lines = []
    for b in sorted(snap["buildings"], key=lambda x: -x.get("population", 0))[:5]:
        building_lines.append(
            f"  Building {b['id']}: pop={b.get('population',0)}, "
            f"damage={graph.get_node_damage(b['id']):.0%}"
        )

    def _dd(a, b):
        return math.sqrt((a["lat"]-b["lat"])**2 + (a["lon"]-b["lon"])**2)

    centre  = {"lat": lat, "lon": lon}
    min_deg = cfg["radius"] / 111_000
    outside_fs   = [f for f in graph.get_fire_stations() if _dd(f, centre) > min_deg]
    outside_hosp = [h for h in graph.get_hospitals()     if _dd(h, centre) > min_deg]

    path = src_fs = dst_hosp = route_data = None
    if outside_fs and outside_hosp:
        src_fs   = min(outside_fs,   key=lambda n: _dd(n, centre))
        dst_hosp = min(outside_hosp, key=lambda n: _dd(n, centre))
        path = graph.shortest_path_astar(src_fs["id"], dst_hosp["id"], avoid_damaged=True)
        if path:
            route_data = {
                "path":     path,
                "dist_km":  graph.path_distance_m(path) / 1000,
                "tt_min":   graph.path_travel_time_s(path) / 60,
                "src_id":   src_fs["id"],
                "dst_id":   dst_hosp["id"],
                "src_node": src_fs,
                "dst_node": dst_hosp,
            }

    if route_data:
        route_hint = (
            f"A* route found: {len(path)} nodes, {route_data['dist_km']:.2f} km, "
            f"~{route_data['tt_min']:.1f} min "
            f"(FS {src_fs['id']} → Hospital {dst_hosp['id']})"
        )
    else:
        staging = graph.find_staging_zones((lat, lon), cfg["radius"])
        route_hint = (
            "NO GROUND ROUTE. Staging zones: " +
            ("; ".join(f"FS {z['id']} at {z['dist_to_epicentre_m']}m" for z in staging)
             if staging else "none found")
        )

    return {
        "map": (
            f"\nDISASTER: {cfg['type'].upper()} | severity={cfg['severity']:.0%} | radius={cfg['radius']}m\n"
            f"Epicentre: ({lat}, {lon})\n"
            f"Nodes affected: {disaster_summary['nodes_affected']} | "
            f"Edges disabled: {disaster_summary['edges_disabled']}\n\n"
            f"ROAD NETWORK (within {cfg['agent_radius']}m):\n"
            + ("\n".join(road_lines) or "  No road nodes.") +
            f"\n\nBUILDINGS (top 5 by population):\n"
            + ("\n".join(building_lines) or "  None.") +
            f"\n  Total population at risk: {snap['population']:,}\n"
        ),
        "status": (
            f"\nSeverity: {cfg['severity']:.0%} | Pop at risk: {disaster_summary['population_at_risk']:,}\n"
            f"Hospitals affected: {disaster_summary['hospitals_affected']} | "
            f"Fire stations affected: {disaster_summary['fire_stations_affected']}\n"
            f"Nearest hospital: id={nearest_h['id']}, ({nearest_h['lat']:.4f},{nearest_h['lon']:.4f})\n"
            f"Nearest fire station: id={nearest_fs['id']}, ({nearest_fs['lat']:.4f},{nearest_fs['lon']:.4f})\n"
        ),
        "task": (
            f"\nTASK: Coordinate {cfg['type']} response at ({lat}, {lon}).\n"
            f"Evacuate {snap['population']:,} people. Route rescue units to hospital.\n"
            f"{route_hint}\n"
        ),
        "snap":       snap,
        "route_data": route_data,
        "src_fs":     src_fs,
        "dst_hosp":   dst_hosp,
    }


# -----------------------------------------------------------------------
# Agent runner
# -----------------------------------------------------------------------

def run_agent(role_prompt: str, world: dict, shared_memory: dict) -> str:
    return call_model(f"""{role_prompt}

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
""")


# -----------------------------------------------------------------------
# Baseline: Dijkstra ignoring damage
# -----------------------------------------------------------------------

def compute_baseline(graph, src_id: int, dst_id: int) -> dict:
    path = graph.shortest_path(src_id, dst_id, avoid_damaged=False)
    if not path:
        return {"path": None, "dist_km": None, "tt_min": None}
    return {
        "path":    path,
        "dist_km": graph.path_distance_m(path) / 1000,
        "tt_min":  graph.path_travel_time_s(path) / 60,
    }


# -----------------------------------------------------------------------
# Single episode
# -----------------------------------------------------------------------

def run_episode(graph, cfg: dict, ep: int, evaluator: "DisasterEvaluator", args):
    print(f"\n{'='*60}")
    print(f"  SIMULATION RUN   (Episode {ep})")
    print(f"{'='*60}")
    print(f"  Type     : {cfg['type'].upper()}")
    print(f"  Location : ({cfg['lat']}, {cfg['lon']})")
    print(f"  Severity : {cfg['severity']:.0%}  |  Radius: {cfg['radius']}m")
    print(f"{'='*60}")

    graph.reset_disaster()
    disaster_summary = graph.apply_disaster(
        cfg["type"], (cfg["lat"], cfg["lon"]), cfg["radius"], cfg["severity"]
    )
    evaluator.set_disaster_summary(disaster_summary, graph)
    world = build_world(graph, cfg, disaster_summary)

    # Baseline
    baseline = {"path": None, "dist_km": None, "tt_min": None}
    if world["src_fs"] and world["dst_hosp"]:
        baseline = compute_baseline(graph, world["src_fs"]["id"], world["dst_hosp"]["id"])
        if baseline["dist_km"]:
            print(f"\n  Baseline (Dijkstra, no damage avoidance): "
                  f"{baseline['dist_km']:.2f} km, {baseline['tt_min']:.1f} min")

    # --- SIMULATION RUN ---
    print("\n--- SIMULATION RUN ---\n")
    shared_memory = {"mapping": "", "risk": "", "resource": "", "routing": ""}
    evaluator.start_step()

    shared_memory["mapping"] = run_agent(MAPPING_PROMPT, world, shared_memory)
    print("[ MAPPING AGENT ]\n" + shared_memory["mapping"] + "\n")

    shared_memory["risk"] = run_agent(RISK_PROMPT, world, shared_memory)
    print("[ RISK AGENT ]\n" + shared_memory["risk"] + "\n")

    shared_memory["resource"] = run_agent(RESOURCE_PROMPT, world, shared_memory)
    print("[ RESOURCE AGENT ]\n" + shared_memory["resource"] + "\n")

    shared_memory["routing"] = run_agent(ROUTING_PROMPT, world, shared_memory)
    print("[ ROUTING AGENT ]\n" + shared_memory["routing"] + "\n")

    evaluator.evaluate_run(shared_memory, world, baseline)

    if not args.no_map:
        map_path = f"simulation_map_ep{ep}.html"
        generate_map(graph, cfg, disaster_summary, world, map_path)
        print(f"  Map saved → {map_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    args = parse_args()

    print("\n  Loading Tokyo graph and building spatial index...")
    graph = HybridTokyoGraph("dataset/tokyo_full_graph_updated.json")
    graph.build_index()

    print(f"\n  Active configuration:")
    print(f"    Disaster : {args.type} | severity={args.severity:.0%} | radius={args.radius}m")
    print(f"    Location : ({args.lat}, {args.lon})")
    print(f"    Episodes : {args.episodes}")

    evaluator = DisasterEvaluator(graph)
    disaster_types = ["earthquake", "flood"]

    for ep in range(1, args.episodes + 1):
        cfg = {
            "lat":          args.lat,
            "lon":          args.lon,
            "type":         disaster_types[(ep-1) % 2] if args.episodes > 1 else args.type,
            "severity":     args.severity,
            "radius":       args.radius,
            "agent_radius": 800,
        }
        run_episode(graph, cfg, ep, evaluator, args)

    evaluator.print_report()


if __name__ == "__main__":
    main()
