from hybrid_spatial_index import HybridTokyoGraph
from visualize import generate_map

# ---- CONFIG ----
cfg = {
    "lat": 35.68,
    "lon": 139.76,
    "type": "earthquake",
    "severity": 0.6,
    "radius": 1000,
    "agent_radius": 2000
}

print("Loading graph...")
graph = HybridTokyoGraph("dataset/tokyo_graph.json")
graph.build_index()

# Fake minimal objects (since we don’t need full run.py pipeline)
disaster_summary = {
    "population_at_risk": 0
}

world = {
    "route_data": None,
    "snap": {"buildings": []}
}

print("Generating map...")
generate_map(graph, cfg, disaster_summary, world, "swarm_map.html")

print("✅ Map saved as swarm_map.html")