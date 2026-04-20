import random
import math
from agent_snn_v3 import SNNAgent, haversine_distance


class SwarmSimulation:
    """
    Swarm simulation — v3 fixes:

    Spawn bug   attempts counter was shared across all agents, so later agents
                exhausted the budget of earlier ones and fell back to unconstrained
                random pairs (up to 8000m). Fixed: reset per-agent.

    Max dist    Added max_start_goal_distance param (default 4000m).
                Agents starting 6000–8000m apart need 200+ steps just to walk
                there — they can never succeed in 200 steps, inflating failure
                counts artificially.  Capping at 4000m keeps all tasks feasible.

    Near-goal   Added _is_near_goal_stuck() — detects agents oscillating within
                300m of goal for 10+ steps and forces a greedy escape that ignores
                taboo/blacklist. This kills the Agent-4-style 43m infinite loop.
    """

    def __init__(self, graph, num_agents=20,
                 min_start_goal_distance=1000,
                 max_start_goal_distance=4000):
        self.graph     = graph
        self.num_agents = num_agents
        self.min_start_goal_distance = min_start_goal_distance
        self.max_start_goal_distance = max_start_goal_distance  # FIX: was hardcoded 8000

        self.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes":  {},
            "pheromones":    {},
        }

        self.PHEROMONE_DECAY   = 0.97
        self.PHEROMONE_DEPOSIT = 0.5

        self.agents = []
        self._spawn_agents()

        self._goal_step = {}

    # ------------------------------------------------------------------ #
    # Spawn
    # ------------------------------------------------------------------ #

    def _spawn_agents(self):
        nodes = list(self.graph._nodes.keys())

        for i in range(self.num_agents):
            # FIX: per-agent attempt budget — old code shared a single counter
            # so agents 7-9 always got zero attempts and fell back to unconstrained pairs
            placed       = False
            max_attempts = 200   # per agent, not total
            attempts     = 0

            while attempts < max_attempts:
                attempts += 1
                start = random.choice(nodes)
                goal  = random.choice(nodes)
                if start == goal:
                    continue
                sd   = self.graph._nodes[start]
                gd   = self.graph._nodes[goal]
                dist = haversine_distance(sd['lat'], sd['lon'], gd['lat'], gd['lon'])
                if self.min_start_goal_distance <= dist <= self.max_start_goal_distance:
                    if self.graph._adj.get(start) and self.graph._adj.get(goal):
                        placed = True
                        break

            if not placed:
                candidates = [n for n in nodes if self.graph._adj.get(n)]
                if len(candidates) >= 2:
                    start, goal = random.sample(candidates, 2)
                else:
                    start, goal = random.sample(nodes, 2)

            agent = SNNAgent(self.graph, start, goal)
            agent.shared_memory = self.shared_memory
            self.agents.append(agent)

            sd   = self.graph._nodes[start]
            gd   = self.graph._nodes[goal]
            dist = haversine_distance(sd['lat'], sd['lon'], gd['lat'], gd['lon'])
            print(f"Agent {i:2d} | Start: {start} → Goal: {goal} | Distance: {dist:.0f}m")

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #

    def run(self, steps=200):
        for t in range(steps):
            print(f"\n{'='*60}")
            print(f"STEP {t}")
            print(f"{'='*60}")

            active_count = 0
            for i, agent in enumerate(self.agents):
                if not agent.alive or agent.reached_goal:
                    continue

                active_count += 1
                self._apply_shared_memory(agent)

                # FIX: near-goal stuck detector — runs before agent.step()
                self._rescue_near_goal_stuck(agent)

                prev_node = agent.current_node
                agent.step()

                if agent.current_node != prev_node:
                    cd  = self.graph._nodes[agent.current_node]
                    gd  = self.graph._nodes[agent.goal_node]
                    dtg = haversine_distance(cd['lat'], cd['lon'], gd['lat'], gd['lon'])
                    status = "ALIVE"
                    if not agent.alive:       status = "DEAD"
                    elif agent.reached_goal:  status = "GOAL!"
                    print(f"  Agent {i:2d}: {prev_node} → {agent.current_node} | "
                          f"Distance to goal: {dtg:6.0f}m | {status}")

                self._update_shared_memory(agent, prev_node)

                if agent.reached_goal:
                    self._deposit_pheromones(agent)
                    self._goal_step[i] = t

            self._evaporate_pheromones()

            active_count_after = sum(
                1 for a in self.agents if a.alive and not a.reached_goal
            )
            print(f"\nActive agents: {active_count_after}")

            if active_count_after == 0:
                print("\nAll agents finished.")
                break

            self._share_information()

    # ------------------------------------------------------------------ #
    # Near-goal stuck rescue
    # ------------------------------------------------------------------ #

    def _rescue_near_goal_stuck(self, agent):
        """
        FIX: Detects the Agent-4 failure mode — oscillating within 300m of goal
        for 10+ steps without closing. When triggered:
          1. Wipes taboo + blacklist (they're blocking the only path forward)
          2. Forces a greedy move: pick the neighbor with minimum dist-to-goal
             ignoring all memory penalties.
        This is safe because we only trigger it when very close AND clearly stuck.
        """
        if agent._last_dist_to_goal is None:
            return
        if agent._last_dist_to_goal > 300:
            return
        if agent._steps_without_progress < 10:
            return

        # Agent is within 300m and hasn't made real progress in 10 steps — rescue it
        neighbors = agent.get_neighbors()
        if not neighbors:
            return

        goal_data = agent.graph._nodes[agent.goal_node]

        # Wipe all memory constraints — the goal is RIGHT THERE
        agent._taboo_reset()
        agent.blacklist.clear()
        agent.snn.reset_momentum()
        agent._steps_without_progress = 0

        # Pure greedy: closest neighbor to goal
        best = min(
            neighbors,
            key=lambda n: haversine_distance(
                agent.graph._nodes[n]['lat'], agent.graph._nodes[n]['lon'],
                goal_data['lat'], goal_data['lon']
            )
        )
        current_pos = agent.graph._nodes[agent.current_node]
        goal_pos    = goal_data
        agent._execute_move(best, current_pos, goal_pos)

    # ------------------------------------------------------------------ #
    # Shared memory helpers
    # ------------------------------------------------------------------ #

    def _apply_shared_memory(self, agent):
        if not hasattr(agent, 'shared_memory') or agent.shared_memory is None:
            agent.shared_memory = self.shared_memory

    def _update_shared_memory(self, agent, prev_node):
        node   = agent.current_node
        hazard = agent.graph.get_node_damage(node)
        if hazard > 0.6:
            self.shared_memory["blocked_nodes"].add(node)
        self.shared_memory["danger_nodes"][node] = hazard

    def _deposit_pheromones(self, agent):
        """
        Deposit strength scales with path quality.
        Shorter paths leave stronger trails — positive feedback toward efficient routes.
        """
        path_len = len(agent.path)
        ph       = self.shared_memory["pheromones"]
        quality  = max(0.5, 1.0 - path_len / 400.0)
        for i, node in enumerate(agent.path):
            position_factor = i / max(path_len - 1, 1)
            strength = self.PHEROMONE_DEPOSIT * position_factor * quality
            ph[node] = min(1.0, ph.get(node, 0.0) + strength)

    def _evaporate_pheromones(self):
        ph = self.shared_memory["pheromones"]
        self.shared_memory["pheromones"] = {
            k: v * self.PHEROMONE_DECAY
            for k, v in ph.items()
            if v * self.PHEROMONE_DECAY > 0.01
        }

    def _share_information(self):
        """Bidirectional max-merge within 500m radius."""
        for i, a in enumerate(self.agents):
            if not a.alive:
                continue
            for j, b in enumerate(self.agents):
                if j <= i or not b.alive:
                    continue
                if self._distance(a.current_node, b.current_node) < 500:
                    all_nodes = (set(a.local_memory["danger_nodes"]) |
                                 set(b.local_memory["danger_nodes"]))
                    for node in all_nodes:
                        ha = a.local_memory["danger_nodes"].get(node, 0.0)
                        hb = b.local_memory["danger_nodes"].get(node, 0.0)
                        merged = max(ha, hb) * 0.95
                        if merged > 0.0:
                            a.local_memory["danger_nodes"][node] = merged
                            b.local_memory["danger_nodes"][node] = merged

    def _distance(self, n1, n2):
        a, b = self.graph._nodes[n1], self.graph._nodes[n2]
        return haversine_distance(a['lat'], a['lon'], b['lat'], b['lon'])

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    def summary(self):
        total   = len(self.agents)
        reached = sum(a.reached_goal for a in self.agents)
        alive   = sum(a.alive and not a.reached_goal for a in self.agents)
        dead    = sum(not a.alive and not a.reached_goal for a in self.agents)

        print("\n" + "=" * 60)
        print("SWARM SUMMARY")
        print("=" * 60)
        print(f"Total agents  : {total}")
        print(f"Reached goal  : {reached} ({100 * reached / total:.1f}%)")
        print(f"Still alive   : {alive}   ({100 * alive / total:.1f}%)")
        print(f"Dead          : {dead}   ({100 * dead / total:.1f}%)")
        print("=" * 60)

        if reached > 0:
            path_lens  = [len(a.path) for a in self.agents if a.reached_goal]
            bl_sizes   = [len(a.blacklist) for a in self.agents if a.reached_goal]
            goal_steps = list(self._goal_step.values())

            print(f"\nPath length   : min={min(path_lens)}  max={max(path_lens)}  "
                  f"mean={sum(path_lens)/len(path_lens):.1f}")
            print(f"Blacklisted   : min={min(bl_sizes)}  max={max(bl_sizes)}  "
                  f"mean={sum(bl_sizes)/len(bl_sizes):.1f} nodes/agent")
            if goal_steps:
                print(f"Goal step     : first={min(goal_steps)}  last={max(goal_steps)}  "
                      f"mean={sum(goal_steps)/len(goal_steps):.1f}")

            print("\nSuccessful paths:")
            for i, agent in enumerate(self.agents):
                if agent.reached_goal:
                    step = self._goal_step.get(i, "?")
                    bl   = len(agent.blacklist)
                    print(f"  Agent {i:2d}: {len(agent.path):4d} steps | "
                          f"reached at step {step} | blacklisted {bl} nodes")

        ph_nodes = len(self.shared_memory["pheromones"])
        print(f"\nPheromone trail: {ph_nodes} nodes marked at run end")
        print("=" * 60)
