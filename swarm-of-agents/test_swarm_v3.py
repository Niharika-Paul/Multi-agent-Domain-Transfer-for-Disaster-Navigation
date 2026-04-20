import sys
import os
import random

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..'))
sys.path.insert(0, root_dir)

from hybrid_spatial_index import HybridTokyoGraph
from swarm_simulation_v3 import SwarmSimulation

script_dir = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH = os.path.abspath(
    os.path.join(script_dir, '../../../../../dataset/tokyo_graph/tokyo_graph.json')
)

DISASTER_CENTRE   = (35.68, 139.76)
DISASTER_RADIUS_M = 2000
DISASTER_SEVERITY = 0.7


def load_fresh_graph():
    """Fresh graph every time — prevents apply_disaster() damage accumulation."""
    graph = HybridTokyoGraph(GRAPH_PATH)
    graph.build_index()
    graph.apply_disaster(
        disaster_type="earthquake",
        centre=DISASTER_CENTRE,
        radius_m=DISASTER_RADIUS_M,
        severity=DISASTER_SEVERITY,
    )
    return graph


def run_single(num_agents=10, steps=200, min_dist=1000, max_dist=4000,
               seed=None, verbose=True):
    """Run one simulation on a fresh graph. Returns summary dict."""
    if seed is not None:
        random.seed(seed)

    graph = load_fresh_graph()
    sim   = SwarmSimulation(
        graph,
        num_agents=num_agents,
        min_start_goal_distance=min_dist,
        max_start_goal_distance=max_dist,   # FIX: cap at 4000m so all tasks are feasible
    )
    sim.run(steps=steps)
    if verbose:
        sim.summary()

    total   = len(sim.agents)
    reached = sum(a.reached_goal for a in sim.agents)
    return {
        "reached":    reached,
        "total":      total,
        "pct":        100 * reached / total,
        "goal_steps": list(sim._goal_step.values()),
        "path_lens":  [len(a.path) for a in sim.agents if a.reached_goal],
    }


def run_experiment(n_runs=5, num_agents=10, steps=200, min_dist=1000, max_dist=4000):
    """
    Run n_runs independent simulations and report aggregate stats.
    Use this for publication-quality numbers — single runs have high variance
    with only 10 agents.
    """
    print(f"\n{'#'*60}")
    print(f"EXPERIMENT: {n_runs} runs x {num_agents} agents x {steps} steps")
    print(f"Distance: {min_dist}–{max_dist}m | "
          f"Disaster: severity={DISASTER_SEVERITY}, radius={DISASTER_RADIUS_M}m")
    print(f"{'#'*60}\n")

    results = []
    for i in range(n_runs):
        print(f"\n{'='*60}")
        print(f"RUN {i+1}/{n_runs}")
        print(f"{'='*60}")
        r = run_single(num_agents=num_agents, steps=steps,
                       min_dist=min_dist, max_dist=max_dist,
                       seed=i*42, verbose=False)
        results.append(r)
        print(f"  --> {r['reached']}/{r['total']} ({r['pct']:.0f}%)")

    pcts      = [r['pct']   for r in results]
    all_steps = [s for r in results for s in r['goal_steps']]
    all_paths = [p for r in results for p in r['path_lens']]

    mean_pct = sum(pcts) / len(pcts)
    variance = sum((p - mean_pct)**2 for p in pcts) / len(pcts)
    std_pct  = variance ** 0.5

    print(f"\n{'#'*60}")
    print(f"AGGREGATE RESULTS ({n_runs} runs)")
    print(f"{'#'*60}")
    print(f"Success rate : {mean_pct:.1f}% +/- {std_pct:.1f}%")
    print(f"Per-run      : {[f'{p:.0f}%' for p in pcts]}")
    if all_steps:
        print(f"Goal step    : mean={sum(all_steps)/len(all_steps):.1f}  "
              f"min={min(all_steps)}  max={max(all_steps)}")
    if all_paths:
        print(f"Path length  : mean={sum(all_paths)/len(all_paths):.1f}  "
              f"min={min(all_paths)}  max={max(all_paths)}")
    print(f"{'#'*60}\n")
    return results


if __name__ == "__main__":
    # Single run — quick sanity check
    run_single(num_agents=10, steps=200, min_dist=1000, max_dist=4000)

    # Multi-run experiment — uncomment for stats:
    # run_experiment(n_runs=5, num_agents=10, steps=200, min_dist=1000, max_dist=4000)
