"""
dataset_connector.py
--------------------
Connects the Tokyo urban graph dataset to the multi-agent disaster
simulation. Provides a clean API that mapping, risk, resource, and
routing agents can call to query the environment.

Dataset summary (tokyo_graph.json):
  - 87,091 nodes  → road_node (76,609), building (10,000),
                     hospital (292), fire_station (190)
  - 197,677 edges → weighted by length (metres), typed by road_type
  - Total modelled population: ~1,206,000 across 9,859 buildings
  - City: Tokyo
"""

import json
import math
from collections import defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# Haversine distance helper
# ---------------------------------------------------------------------------

def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Return distance in metres between two lat/lon points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# TokyoGraph  –  main dataset interface
# ---------------------------------------------------------------------------

class TokyoGraph:
    """
    Loads the Tokyo urban graph and exposes query methods used by agents.

    Usage:
        graph = TokyoGraph("tokyo_graph.json")

        # --- Mapping agent ---
        nodes = graph.get_nodes_in_radius(35.68, 139.76, radius_m=1000)

        # --- Risk agent ---
        pop   = graph.get_population_in_radius(35.68, 139.76, radius_m=500)
        graph.apply_disaster(disaster_type="earthquake",
                             centre=(35.68, 139.76), radius_m=2000,
                             severity=0.8)

        # --- Resource agent ---
        hospitals   = graph.get_hospitals()
        fire_stations = graph.get_fire_stations()
        nearest_h   = graph.nearest_facility("hospital", lat, lon)

        # --- Routing agent ---
        path = graph.shortest_path(node_id_from, node_id_to)
    """

    # Road types treated as passable during disasters (rough priority order)
    _ROAD_PRIORITY = {
        "motorway": 1, "trunk": 2, "primary": 3,
        "secondary": 4, "tertiary": 5, "residential": 6,
        "unclassified": 7,
    }

    def __init__(self, json_path: str):
        with open(json_path) as f:
            raw = json.load(f)

        self.city: str = raw.get("city", "Tokyo")
        self._nodes: dict = {}        # id → node dict
        self._adj: dict = defaultdict(list)  # id → [(neighbour_id, edge)]
        self._disabled_edges: set = set()    # frozenset({from, to}) for blocked roads
        self._node_damage: dict = {}  # id → damage factor 0..1

        # Index nodes
        for node in raw["nodes"]:
            self._nodes[node["id"]] = node

        # Index adjacency list
        for edge in raw["edges"]:
            frm, to = edge["from"], edge["to"]
            self._adj[frm].append((to, edge))
            self._adj[to].append((frm, edge))   # treat as undirected

        # Spatial index: lat/lon buckets (0.01° ≈ 1 km) for fast radius queries
        self._bucket: dict = defaultdict(list)
        for nid, node in self._nodes.items():
            bkey = (round(node["lat"], 2), round(node["lon"], 2))
            self._bucket[bkey].append(nid)

    # -----------------------------------------------------------------------
    # Spatial queries  (used by Mapping Agent + Risk Agent)
    # -----------------------------------------------------------------------

    def get_nodes_in_radius(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        node_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Return all nodes within radius_m metres of (lat, lon).
        Optionally filter by node_type: 'road_node', 'building',
        'hospital', 'fire_station'.
        """
        # Expand bucket search area (1° lat ≈ 111 km)
        deg_spread = (radius_m / 111_000) + 0.02
        candidates = set()
        lat0, lon0 = round(lat, 2), round(lon, 2)
        steps = math.ceil(deg_spread / 0.01) + 1

        for dlat in range(-steps, steps + 1):
            for dlon in range(-steps, steps + 1):
                bkey = (round(lat0 + dlat * 0.01, 2),
                        round(lon0 + dlon * 0.01, 2))
                candidates.update(self._bucket.get(bkey, []))

        results = []
        for nid in candidates:
            node = self._nodes[nid]
            if node_type and node.get("type", "road_node") != node_type:
                continue
            if _haversine_m(lat, lon, node["lat"], node["lon"]) <= radius_m:
                results.append(node)
        return results

    def get_population_in_radius(self, lat: float, lon: float, radius_m: float) -> int:
        """Total population of buildings within radius_m metres."""
        buildings = self.get_nodes_in_radius(lat, lon, radius_m, node_type="building")
        return sum(b.get("population", 0) for b in buildings)

    def nearest_road_node(self, lat: float, lon: float) -> Optional[dict]:
        """
        Return the nearest *connected* road node to a coordinate.
        Use this to find routing entry/exit points for hospitals and
        fire stations (which are not directly in the road graph).
        """
        for search_r in [500, 2000, 10_000]:
            candidates = self.get_nodes_in_radius(lat, lon, search_r, node_type="road_node")
            # Only return nodes that are actually in the adjacency list
            connected = [n for n in candidates if n["id"] in self._adj]
            if connected:
                return min(connected,
                           key=lambda n: _haversine_m(lat, lon, n["lat"], n["lon"]))
        return None

    def nearest_node(self, lat: float, lon: float) -> dict:
        """Return the nearest node (any type) to a given coordinate."""
        best, best_d = None, float("inf")
        for search_r in [500, 2000, 10000]:
            candidates = self.get_nodes_in_radius(lat, lon, search_r)
            if candidates:
                for node in candidates:
                    d = _haversine_m(lat, lon, node["lat"], node["lon"])
                    if d < best_d:
                        best, best_d = node, d
                break
        return best

    # -----------------------------------------------------------------------
    # Infrastructure queries  (used by Resource Agent)
    # -----------------------------------------------------------------------

    def get_hospitals(self) -> list[dict]:
        """All hospital nodes."""
        return [n for n in self._nodes.values() if n.get("type") == "hospital"]

    def get_fire_stations(self) -> list[dict]:
        """All fire station nodes."""
        return [n for n in self._nodes.values() if n.get("type") == "fire_station"]

    def get_buildings(self) -> list[dict]:
        """All building nodes (with population)."""
        return [n for n in self._nodes.values() if n.get("type") == "building"]

    def nearest_facility(self, facility_type: str, lat: float, lon: float) -> Optional[dict]:
        """
        Return the nearest hospital or fire_station to a coordinate.
        facility_type: 'hospital' | 'fire_station'
        """
        facilities = (self.get_hospitals() if facility_type == "hospital"
                      else self.get_fire_stations())
        if not facilities:
            return None
        return min(facilities,
                   key=lambda n: _haversine_m(lat, lon, n["lat"], n["lon"]))

    def facilities_in_radius(
        self, facility_type: str, lat: float, lon: float, radius_m: float
    ) -> list[dict]:
        """All facilities of a given type within radius_m."""
        return self.get_nodes_in_radius(lat, lon, radius_m, node_type=facility_type)

    # -----------------------------------------------------------------------
    # Disaster simulation  (used by Risk Agent)
    # -----------------------------------------------------------------------

    def apply_disaster(
        self,
        disaster_type: str,
        centre: tuple[float, float],
        radius_m: float,
        severity: float = 0.5,
    ) -> dict:
        """
        Mark nodes and edges in the affected zone as damaged.

        severity: 0.0 (no damage) → 1.0 (total destruction)

        Returns a summary dict for the Risk Agent to broadcast.
        """
        lat, lon = centre
        affected_nodes = self.get_nodes_in_radius(lat, lon, radius_m)
        affected_ids = {n["id"] for n in affected_nodes}

        # Damage nodes (proportional to severity, attenuated by distance)
        for node in affected_nodes:
            dist = _haversine_m(lat, lon, node["lat"], node["lon"])
            attenuation = 1 - (dist / radius_m) * 0.5   # closer = more damage
            damage = severity * attenuation
            self._node_damage[node["id"]] = min(damage, 1.0)

        # Disable edges where both endpoints are in the affected zone
        disabled_count = 0
        for nid in affected_ids:
            for neighbour_id, edge in self._adj[nid]:
                if neighbour_id in affected_ids:
                    key = frozenset({nid, neighbour_id})
                    # Disable high-severity bridges / major roads more readily
                    road_priority = self._ROAD_PRIORITY.get(
                        str(edge.get("road_type", "")), 7
                    )
                    disable_threshold = 0.3 + (road_priority / 10) * 0.4
                    if severity >= disable_threshold:
                        self._disabled_edges.add(key)
                        disabled_count += 1

        pop_at_risk = self.get_population_in_radius(lat, lon, radius_m)
        hospitals_affected = len(self.facilities_in_radius("hospital", lat, lon, radius_m))
        fire_affected = len(self.facilities_in_radius("fire_station", lat, lon, radius_m))

        return {
            "disaster_type": disaster_type,
            "centre": {"lat": lat, "lon": lon},
            "radius_m": radius_m,
            "severity": severity,
            "nodes_affected": len(affected_nodes),
            "edges_disabled": disabled_count,
            "population_at_risk": pop_at_risk,
            "hospitals_affected": hospitals_affected,
            "fire_stations_affected": fire_affected,
        }

    def reset_disaster(self):
        """Clear all damage state (use between simulation runs)."""
        self._disabled_edges.clear()
        self._node_damage.clear()

    def is_edge_passable(self, node_from_id: int, node_to_id: int) -> bool:
        """Check whether a road edge is currently passable."""
        return frozenset({node_from_id, node_to_id}) not in self._disabled_edges

    def get_node_damage(self, node_id: int) -> float:
        """Return damage factor (0.0 = intact, 1.0 = destroyed)."""
        return self._node_damage.get(node_id, 0.0)

    # -----------------------------------------------------------------------
    # Routing  (used by Routing Agent)
    # -----------------------------------------------------------------------

    def shortest_path(
        self, from_id: int, to_id: int, avoid_damaged: bool = True
    ) -> Optional[list[int]]:
        """
        Dijkstra shortest path between two node IDs.
        Returns list of node IDs or None if no path found.

        avoid_damaged: skip disabled edges (disaster-aware routing)
        """
        import heapq

        if from_id not in self._nodes or to_id not in self._nodes:
            return None

        dist = {from_id: 0.0}
        prev = {}
        heap = [(0.0, from_id)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")):
                continue
            if u == to_id:
                break
            for v, edge in self._adj[u]:
                if avoid_damaged and not self.is_edge_passable(u, v):
                    continue
                weight = edge["length"]
                # Penalise damaged nodes along the route
                weight *= (1 + self._node_damage.get(v, 0.0) * 2)
                nd = d + weight
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

        if to_id not in prev and from_id != to_id:
            return None   # unreachable

        path = []
        cur = to_id
        while cur != from_id:
            path.append(cur)
            cur = prev[cur]
        path.append(from_id)
        return list(reversed(path))

    def path_distance_m(self, path: list[int]) -> float:
        """Total length of a path (list of node IDs) in metres."""
        total = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            for neighbour, edge in self._adj[u]:
                if neighbour == v:
                    total += edge["length"]
                    break
        return total

    # -----------------------------------------------------------------------
    # Snapshot for agent shared memory
    # -----------------------------------------------------------------------

    def snapshot_local(self, lat: float, lon: float, radius_m: float) -> dict:
        """
        Return a compact snapshot of the environment around a coordinate.
        Each agent calls this to populate its local observation.
        """
        nodes = self.get_nodes_in_radius(lat, lon, radius_m)
        return {
            "centre": {"lat": lat, "lon": lon},
            "radius_m": radius_m,
            "road_nodes": [n for n in nodes if n.get("type", "road_node") == "road_node"],
            "buildings": [n for n in nodes if n.get("type") == "building"],
            "hospitals": [n for n in nodes if n.get("type") == "hospital"],
            "fire_stations": [n for n in nodes if n.get("type") == "fire_station"],
            "population": sum(n.get("population", 0) for n in nodes if n.get("type") == "building"),
            "damaged_nodes": [nid for nid in self._node_damage
                              if any(n["id"] == nid for n in nodes)],
        }

    def __repr__(self):
        return (f"<TokyoGraph city={self.city!r} "
                f"nodes={len(self._nodes):,} edges={sum(len(v) for v in self._adj.values())//2:,}>")
