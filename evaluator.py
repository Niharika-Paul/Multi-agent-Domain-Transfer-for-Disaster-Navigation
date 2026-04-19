"""
evaluator.py — Quantitative Disaster Simulation Metrics
---------------------------------------------------------
Metrics:
  1. Path Success Rate     — successful_routes / total_routes
  2. Path Efficiency       — straight_line_distance / actual_path_distance
  3. Network Robustness    — remaining_active_edges / total_edges
  4. Coverage              — reachable_buildings / total_buildings (sampled)
  5. Avg Response Time     — total_wall_clock_time / steps
  + Baseline comparison    — agent path vs Dijkstra (no damage avoidance)
"""

import re
import time
import math
from dataclasses import dataclass, field
from typing import Optional


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


@dataclass
class RunResult:
    episode:           int
    success:           bool
    path_efficiency:   float   # 0–1, higher = more direct route
    network_robustness:float   # 0–1, fraction of edges still active
    coverage:          float   # 0–1, fraction of buildings reachable
    avg_response_time: float   # seconds
    agent_dist_km:     Optional[float]
    baseline_dist_km:  Optional[float]
    improvement_pct:   Optional[float]  # vs baseline, positive = agent better
    notes:             list[str] = field(default_factory=list)


class DisasterEvaluator:
    """
    Accumulates metrics across multiple simulation episodes and prints
    a structured report.

    Usage in run.py:
        evaluator = DisasterEvaluator(graph)
        evaluator.set_disaster_summary(summary, graph)   # after apply_disaster
        evaluator.start_step()                           # before agents run
        evaluator.evaluate_run(shared_memory, world, baseline)
        evaluator.print_report()
    """

    def __init__(self, graph):
        self._graph = graph
        self._results: list[RunResult] = []
        self._step_start = time.time()
        self._disaster_summary: dict = {}
        self._total_edges: int = 0

    def set_disaster_summary(self, summary: dict, graph):
        self._disaster_summary = summary
        # Count total edges once
        self._total_edges = sum(len(v) for v in graph._adj.values()) // 2

    def start_step(self):
        self._step_start = time.time()

    # -----------------------------------------------------------------------
    # Main evaluation entry point
    # -----------------------------------------------------------------------

    def evaluate_run(self, shared_memory: dict, world: dict, baseline: dict):
        elapsed = time.time() - self._step_start
        routing_text = shared_memory.get("routing", "") or ""
        episode = len(self._results) + 1

        # 1. Path success
        success, agent_path_nodes = self._parse_route_success(routing_text)

        # 2. Path efficiency  (straight-line / actual)
        efficiency = self._compute_efficiency(world, success)

        # 3. Network robustness
        robustness = self._compute_robustness()

        # 4. Coverage
        coverage = self._compute_coverage()

        # 5. Baseline comparison
        agent_km   = world["route_data"]["dist_km"] if world.get("route_data") else None
        baseline_km = baseline.get("dist_km")
        improvement = None
        if agent_km and baseline_km and baseline_km > 0:
            improvement = ((baseline_km - agent_km) / baseline_km) * 100

        notes = []
        if not success:
            notes.append("No ground route found by routing agent.")
        if improvement is not None:
            sign = "+" if improvement >= 0 else ""
            notes.append(f"Agent route {sign}{improvement:.1f}% vs baseline Dijkstra.")

        result = RunResult(
            episode=episode,
            success=success,
            path_efficiency=efficiency,
            network_robustness=robustness,
            coverage=coverage,
            avg_response_time=elapsed,
            agent_dist_km=agent_km,
            baseline_dist_km=baseline_km,
            improvement_pct=improvement,
            notes=notes,
        )
        self._results.append(result)
        return result

    # -----------------------------------------------------------------------
    # Metric helpers
    # -----------------------------------------------------------------------

    def _parse_route_success(self, routing_text: str):
        if not routing_text.strip():
            return False, 0
        if "NO SAFE PATH" in routing_text.upper():
            return False, 0
        arrows = routing_text.count("->")
        return (arrows > 0), arrows + 1

    def _compute_efficiency(self, world: dict, success: bool) -> float:
        """straight-line distance / actual path distance (capped at 1.0)."""
        if not success or not world.get("route_data"):
            return 0.0
        rd = world["route_data"]
        if not rd or not rd.get("dist_km") or rd["dist_km"] == 0:
            return 0.0
        src = rd["src_node"]
        dst = rd["dst_node"]
        straight_km = _haversine_m(
            src["lat"], src["lon"], dst["lat"], dst["lon"]
        ) / 1000
        return round(min(straight_km / rd["dist_km"], 1.0), 3)

    def _compute_robustness(self) -> float:
        """Fraction of edges remaining active after disaster."""
        if self._total_edges == 0:
            return 1.0
        disabled = len(self._graph._disabled_edges)
        return round(max(0.0, (self._total_edges - disabled) / self._total_edges), 3)

    def _compute_coverage(self, sample_n: int = 200) -> float:
        """
        Fraction of sampled buildings reachable from any fire station
        via the current (post-disaster) road network using BFS.
        Uses a random sample for speed.
        """
        import random
        buildings = self._graph.get_buildings()
        if not buildings:
            return 0.0

        sample = random.sample(buildings, min(sample_n, len(buildings)))

        # BFS from all fire stations simultaneously
        fire_stations = self._graph.get_fire_stations()
        if not fire_stations:
            return 0.0

        visited = set()
        frontier = [fs["id"] for fs in fire_stations]
        for nid in frontier:
            visited.add(nid)

        # Bounded BFS (max 5000 nodes to keep it fast)
        steps = 0
        while frontier and steps < 5000:
            next_f = []
            for nid in frontier:
                for nb_id, _ in self._graph._adj.get(nid, []):
                    if nb_id not in visited and self._graph.is_edge_passable(nid, nb_id):
                        visited.add(nb_id)
                        next_f.append(nb_id)
            frontier = next_f
            steps += len(next_f)

        reachable = sum(1 for b in sample if b["id"] in visited)
        return round(reachable / len(sample), 3)

    # -----------------------------------------------------------------------
    # Aggregates
    # -----------------------------------------------------------------------

    @property
    def success_rate(self) -> float:
        if not self._results: return 0.0
        return round(sum(1 for r in self._results if r.success) / len(self._results), 3)

    @property
    def mean_efficiency(self) -> float:
        if not self._results: return 0.0
        return round(sum(r.path_efficiency for r in self._results) / len(self._results), 3)

    @property
    def mean_robustness(self) -> float:
        if not self._results: return 0.0
        return round(sum(r.network_robustness for r in self._results) / len(self._results), 3)

    @property
    def mean_coverage(self) -> float:
        if not self._results: return 0.0
        return round(sum(r.coverage for r in self._results) / len(self._results), 3)

    @property
    def mean_response_time(self) -> float:
        if not self._results: return 0.0
        return round(sum(r.avg_response_time for r in self._results) / len(self._results), 2)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------

    def print_report(self):
        print("\n" + "=" * 56)
        print("  SYSTEM METRICS")
        print("=" * 56)

        if not self._results:
            print("  No runs recorded.")
            return

        r = self._results[-1]  # latest episode for per-run metrics

        print(f"  ✔ Path Success Rate  : {self.success_rate:.0%}")
        print(f"  ✔ Network Robustness : {self.mean_robustness:.0%}  "
              f"({int(self.mean_robustness * self._total_edges):,} / "
              f"{self._total_edges:,} edges active)")
        print(f"  ✔ Coverage           : {self.mean_coverage:.0%}  "
              f"(buildings reachable from fire stations)")
        print(f"  ✔ Avg Travel Time    : {self.mean_response_time:.1f}s / run")
        print(f"  ✔ Avg Path Efficiency: {self.mean_efficiency:.0%}  "
              f"(straight-line / actual path)")

        if len(self._results) > 1:
            print(f"\n  Episodes run: {len(self._results)}")

        print()

        # Per-episode breakdown
        if len(self._results) > 1:
            print(f"  {'Ep':<4} {'OK':<4} {'Effic':<8} {'Robust':<9} "
                  f"{'Cover':<8} {'Time':<8} {'Agent km':<10} {'Base km':<10} Notes")
            print("  " + "-" * 76)
            for r in self._results:
                ok   = "✓" if r.success else "✗"
                note = "; ".join(r.notes) if r.notes else "-"
                akm  = f"{r.agent_dist_km:.2f}" if r.agent_dist_km else "-"
                bkm  = f"{r.baseline_dist_km:.2f}" if r.baseline_dist_km else "-"
                print(f"  {r.episode:<4} {ok:<4} {r.path_efficiency:<8.0%} "
                      f"{r.network_robustness:<9.0%} {r.coverage:<8.0%} "
                      f"{r.avg_response_time:<8.1f} {akm:<10} {bkm:<10} {note}")
        else:
            r = self._results[0]
            if r.agent_dist_km and r.baseline_dist_km:
                print(f"  Agent route  : {r.agent_dist_km:.2f} km (damage-aware A*)")
                print(f"  Baseline     : {r.baseline_dist_km:.2f} km (Dijkstra, no damage)")
                if r.improvement_pct is not None:
                    sign = "+" if r.improvement_pct >= 0 else ""
                    print(f"  Difference   : {sign}{r.improvement_pct:.1f}%")
            for note in r.notes:
                print(f"  {note}")

        print("=" * 56)

    def as_dict(self) -> dict:
        return {
            "episodes": len(self._results),
            "success_rate": self.success_rate,
            "mean_path_efficiency": self.mean_efficiency,
            "mean_network_robustness": self.mean_robustness,
            "mean_coverage": self.mean_coverage,
            "mean_response_time_s": self.mean_response_time,
            "per_episode": [
                {
                    "episode": r.episode,
                    "success": r.success,
                    "path_efficiency": r.path_efficiency,
                    "network_robustness": r.network_robustness,
                    "coverage": r.coverage,
                    "response_time_s": round(r.avg_response_time, 2),
                    "agent_dist_km": r.agent_dist_km,
                    "baseline_dist_km": r.baseline_dist_km,
                    "improvement_pct": r.improvement_pct,
                }
                for r in self._results
            ],
        }
