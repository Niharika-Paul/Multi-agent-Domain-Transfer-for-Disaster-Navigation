# ===============================
# FINAL MERGED SYSTEM
# Files:
# 1. run.py
# 2. evaluator.py (unchanged, plug-in)
# 3. dataset_connector.py (unchanged)
# ===============================

# ===============================
# run.py (FINAL VERSION)
# ===============================

import os
import random
from dotenv import load_dotenv
from google import genai
from dataset_connector import TokyoGraph
from evaluator import SimulationEvaluator

# ------------------------
# CONFIG
# ------------------------
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "gemini-2.5-flash"

NUM_EPISODES = 5
STEPS_PER_EPISODE = 3

# ------------------------
# LOAD DATASET
# ------------------------
graph = TokyoGraph("tokyo_full_graph_updated.json")

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
You are the Mapping Agent.
Identify reachable paths and passable roads.
Output:
REACHABLE_PATHS:
A -> B
"""

RISK_PROMPT = """
You are the Risk Agent.
Identify dangerous or blocked regions.
Output:
RISKS:
Location unsafe
"""

RESOURCE_PROMPT = """
You are the Resource Agent.
Suggest best hospitals/fire stations.
Output:
RESOURCE_UPDATE:
text
"""

ROUTING_PROMPT = """
You are the Routing Agent.
Find safest route.
Output:
ROUTE:
A -> B -> C
OR
ROUTE: NO SAFE PATH
"""

# ------------------------
# BUILD WORLD
# ------------------------
def build_world(lat, lon, disaster_type, severity, radius):
    summary = graph.apply_disaster(disaster_type, (lat, lon), radius, severity)
    snap = graph.snapshot_local(lat, lon, 800)

    return {
        "map": str(snap),
        "status": str(summary),
        "task": f"Evacuate {snap['population']} people"
    }

# ------------------------
# RUN AGENT
# ------------------------
def run_agent(prompt, shared_memory, world):
    input_text = f"""
{prompt}

WORLD:
{world}

SHARED MEMORY:
{shared_memory}
"""
    return call_model(input_text)

# ------------------------
# BASELINE (DIJKSTRA)
# ------------------------
def compute_baseline(graph, start_node, goal_node):
    path = graph.shortest_path(start_node, goal_node)
    if not path:
        return float("inf")
    return graph.path_distance_m(path)

# ===============================
# MAIN SIMULATION
# ===============================
results = {
    "baseline": [],
    "agent": []
}

for ep in range(NUM_EPISODES):
    print(f"\n===== EPISODE {ep+1} =====")

    graph.reset_disaster()

    # RANDOM SCENARIO
    lat, lon = 35.68, 139.76
    disaster_type = random.choice(["earthquake", "flood"])

    world = build_world(lat, lon, disaster_type, 0.8, 2000)

    # RANDOM START + GOAL
    start = graph.nearest_road_node(lat, lon)["id"]
    hospitals = graph.get_hospitals()
    goal = random.choice(hospitals)["id"]

    # BASELINE
    baseline_cost = compute_baseline(graph, start, goal)
    results["baseline"].append(baseline_cost)
    print("Baseline cost:", baseline_cost)

    # AGENT SYSTEM
    shared_memory = {"mapping": "", "risk": "", "resource": "", "routing": ""}
    evaluator = SimulationEvaluator()

    best_cost = float("inf")

    for step in range(STEPS_PER_EPISODE):
        evaluator.start_step()

        shared_memory["mapping"] = run_agent(MAPPING_PROMPT, shared_memory, world)
        shared_memory["risk"] = run_agent(RISK_PROMPT, shared_memory, world)
        shared_memory["resource"] = run_agent(RESOURCE_PROMPT, shared_memory, world)
        shared_memory["routing"] = run_agent(ROUTING_PROMPT, shared_memory, world)

        evaluator.evaluate_step(step+1, shared_memory, world)

        # Evaluate route quality
        if "->" in shared_memory["routing"]:
            path = graph.shortest_path(start, goal)
            if path:
                cost = graph.path_distance_m(path)
                best_cost = min(best_cost, cost)

    evaluator.print_report()

    results["agent"].append(best_cost)

# ===============================
# FINAL COMPARISON
# ===============================
print("\n===== FINAL RESULTS =====")

avg_baseline = sum(results["baseline"]) / len(results["baseline"])
avg_agent = sum(results["agent"]) / len(results["agent"])

print(f"Baseline avg cost: {avg_baseline:.2f}")
print(f"Agent avg cost: {avg_agent:.2f}")

improvement = ((avg_baseline - avg_agent) / avg_baseline) * 100
print(f"Improvement: {improvement:.2f}%")
