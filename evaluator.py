"""
evaluator.py
------------
Scores each simulation run from run.py across three metrics from the
project spec: success rate, coordination efficiency, and response time.

Drop-in usage — call evaluate_step() inside your existing loop in run.py,
then print_report() at the end.

Quick integration (add 3 lines to run.py):
    from evaluator import SimulationEvaluator
    evaluator = SimulationEvaluator()          # before the loop

    # inside the loop, after all four agents run:
    evaluator.evaluate_step(step + 1, shared_memory, world)

    # after the loop:
    evaluator.print_report()
"""

import re
import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    step: int
    success: bool           # routing agent produced a valid path
    coordination: float     # 0.0 – 1.0: how well agents built on each other
    elapsed_seconds: float  # wall-clock time for this step
    route_hops: int         # number of nodes in the route (0 if none)
    risks_identified: int   # count of distinct risks flagged
    resources_recommended: int  # count of resource actions suggested
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

class SimulationEvaluator:
    """
    Evaluates simulation steps and accumulates metrics across a run.

    Metrics
    -------
    success_rate        — fraction of steps where routing found a valid path
    coordination_score  — average inter-agent alignment score (0–1)
    response_time       — mean wall-clock seconds per step; also tracks
                          step_to_resolution (first step a path was found)
    """

    def __init__(self):
        self._results: list[StepResult] = []
        self._step_start: float = time.time()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def start_step(self):
        """Call just before running the four agents in a step."""
        self._step_start = time.time()

    def evaluate_step(
        self,
        step: int,
        shared_memory: dict,
        world: dict,
    ) -> StepResult:
        """
        Score one completed simulation step.

        Parameters
        ----------
        step          : 1-based step number
        shared_memory : the dict with keys mapping, risk, resource, routing
        world         : the world dict from run.py (used for context checks)

        Returns the StepResult (also stored internally).
        """
        elapsed = time.time() - self._step_start

        routing_text   = shared_memory.get("routing",  "") or ""
        mapping_text   = shared_memory.get("mapping",  "") or ""
        risk_text      = shared_memory.get("risk",     "") or ""
        resource_text  = shared_memory.get("resource", "") or ""

        success, route_hops, notes = self._score_success(routing_text)
        coordination              = self._score_coordination(
            mapping_text, risk_text, resource_text, routing_text
        )
        risks       = self._count_risks(risk_text)
        resources   = self._count_resources(resource_text)

        result = StepResult(
            step=step,
            success=success,
            coordination=coordination,
            elapsed_seconds=elapsed,
            route_hops=route_hops,
            risks_identified=risks,
            resources_recommended=resources,
            notes=notes,
        )
        self._results.append(result)
        return result

    # -----------------------------------------------------------------------
    # Metric 1 — Success rate
    # -----------------------------------------------------------------------

    def _score_success(self, routing_text: str) -> tuple[bool, int, list[str]]:
        """
        A step is successful if the routing agent output a valid path
        (contains at least one '->' that is NOT the NO SAFE PATH sentinel).

        Returns (success, route_hop_count, notes_list).
        """
        notes = []

        if not routing_text.strip():
            notes.append("Routing agent produced no output.")
            return False, 0, notes

        if "NO SAFE PATH" in routing_text.upper():
            notes.append("Routing agent declared no safe path.")
            return False, 0, notes

        # Count '->' occurrences as a proxy for path hops
        arrows = routing_text.count("->")
        if arrows == 0:
            notes.append("Routing output lacks '->' separators — path unclear.")
            return False, 0, notes

        hop_count = arrows + 1   # A->B->C has 2 arrows, 3 nodes
        notes.append(f"Valid route found: ~{hop_count} stops.")
        return True, hop_count, notes

    # -----------------------------------------------------------------------
    # Metric 2 — Coordination efficiency
    # -----------------------------------------------------------------------

    def _score_coordination(
        self,
        mapping: str,
        risk: str,
        resource: str,
        routing: str,
    ) -> float:
        """
        Measures how well agents build on each other's outputs.
        Score is the average of four binary sub-checks (0.0 – 1.0).

        Sub-checks
        ----------
        1. Mapping output is non-trivial (contains node references or paths)
        2. Risk output references something the mapping agent found
        3. Resource output references something from risk or mapping
        4. Routing output references nodes or locations from earlier agents
        """
        scores = []

        # 1. Mapping is substantive
        mapping_has_nodes = bool(
            re.search(r'\d{6,}', mapping) or      # long node IDs
            re.search(r'\d+\.\d+', mapping) or     # lat/lon decimals
            '->' in mapping
        )
        scores.append(1.0 if mapping_has_nodes else 0.0)

        # 2. Risk references mapping content
        # Extract any numbers/locations from mapping and check if risk mentions them
        mapping_tokens = set(re.findall(r'\b\d{4,}\b', mapping))
        mapping_tokens |= set(re.findall(r'\b[A-Z][a-z]+(?:_[A-Z][a-z]+)+\b', mapping))
        mapping_tokens |= set(re.findall(r'\b(?:hospital|fire.station|bridge|road)\b',
                                         mapping, re.IGNORECASE))
        risk_references_mapping = any(tok.lower() in risk.lower() for tok in mapping_tokens) \
            if mapping_tokens else bool(risk.strip())
        scores.append(1.0 if risk_references_mapping else 0.0)

        # 3. Resource references risk or mapping content
        risk_tokens = set(re.findall(r'\b\d{4,}\b', risk))
        risk_tokens |= set(re.findall(r'\b(?:hospital|fire.station|overcrowd|damage|blocked)\b',
                                       risk, re.IGNORECASE))
        combined_tokens = mapping_tokens | risk_tokens
        resource_references_upstream = any(
            tok.lower() in resource.lower() for tok in combined_tokens
        ) if combined_tokens else bool(resource.strip())
        scores.append(1.0 if resource_references_upstream else 0.0)

        # 4. Routing references upstream content (nodes, locations, or risk terms)
        routing_references_upstream = any(
            tok.lower() in routing.lower() for tok in combined_tokens
        ) if combined_tokens else bool(routing.strip())
        scores.append(1.0 if routing_references_upstream else 0.5)  # 0.5 if routing ran at all

        return round(sum(scores) / len(scores), 2)

    # -----------------------------------------------------------------------
    # Metric 3 — Response time helpers
    # -----------------------------------------------------------------------

    def _count_risks(self, risk_text: str) -> int:
        """Count distinct risk items (lines with 'unsafe', 'blocked', 'damaged', etc.)."""
        lines = [l.strip() for l in risk_text.splitlines() if l.strip()]
        risk_keywords = re.compile(
            r'\b(unsafe|blocked|damaged|compromised|overcrowd|flood|collapse|danger|risk)\b',
            re.IGNORECASE
        )
        return sum(1 for l in lines if risk_keywords.search(l))

    def _count_resources(self, resource_text: str) -> int:
        """Count distinct resource recommendations."""
        lines = [l.strip() for l in resource_text.splitlines() if l.strip()]
        action_keywords = re.compile(
            r'\b(redirect|deploy|use|send|allocate|divert|assign|recommend|suggest|avoid)\b',
            re.IGNORECASE
        )
        return sum(1 for l in lines if action_keywords.search(l))

    # -----------------------------------------------------------------------
    # Aggregate report
    # -----------------------------------------------------------------------

    @property
    def success_rate(self) -> float:
        if not self._results:
            return 0.0
        return round(sum(1 for r in self._results if r.success) / len(self._results), 2)

    @property
    def mean_coordination(self) -> float:
        if not self._results:
            return 0.0
        return round(sum(r.coordination for r in self._results) / len(self._results), 2)

    @property
    def mean_response_time(self) -> float:
        if not self._results:
            return 0.0
        return round(sum(r.elapsed_seconds for r in self._results) / len(self._results), 2)

    @property
    def step_to_resolution(self) -> int:
        """First step where a valid route was found, or -1 if never."""
        for r in self._results:
            if r.success:
                return r.step
        return -1

    def print_report(self):
        """Print a formatted summary of the simulation run."""
        print("\n" + "=" * 52)
        print("  SIMULATION EVALUATION REPORT")
        print("=" * 52)

        if not self._results:
            print("  No steps recorded.")
            return

        print(f"  Steps run           : {len(self._results)}")
        print(f"  Step to resolution  : "
              f"{'Step ' + str(self.step_to_resolution) if self.step_to_resolution > 0 else 'Never'}")
        print()
        print(f"  ✦ Success rate      : {self.success_rate:.0%}")
        print(f"  ✦ Coordination score: {self.mean_coordination:.0%}")
        print(f"  ✦ Mean response time: {self.mean_response_time:.2f}s / step")
        print()
        print("  Per-step breakdown:")
        print(f"  {'Step':<6} {'Success':<10} {'Coord':<8} {'Time(s)':<10} "
              f"{'Risks':<8} {'Resources':<10} Notes")
        print("  " + "-" * 68)
        for r in self._results:
            success_str = "✓" if r.success else "✗"
            notes_str   = "; ".join(r.notes) if r.notes else "-"
            print(f"  {r.step:<6} {success_str:<10} {r.coordination:<8.0%} "
                  f"{r.elapsed_seconds:<10.2f} {r.risks_identified:<8} "
                  f"{r.resources_recommended:<10} {notes_str}")

        print("=" * 52)

    def as_dict(self) -> dict:
        """Return metrics as a plain dict (useful for logging or comparison)."""
        return {
            "steps_run": len(self._results),
            "success_rate": self.success_rate,
            "mean_coordination_score": self.mean_coordination,
            "mean_response_time_s": self.mean_response_time,
            "step_to_resolution": self.step_to_resolution,
            "per_step": [
                {
                    "step": r.step,
                    "success": r.success,
                    "coordination": r.coordination,
                    "elapsed_s": round(r.elapsed_seconds, 3),
                    "risks_identified": r.risks_identified,
                    "resources_recommended": r.resources_recommended,
                }
                for r in self._results
            ],
        }
