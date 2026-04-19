"""
visualize.py — Interactive Folium Map for Disaster Simulation
-------------------------------------------------------------
Generates an HTML map with:
  - Road network (green=safe, yellow=damaged, red=blocked)
  - Buildings coloured by damage level
  - Hospital and fire station markers
  - Disaster epicentre + radius circle
  - Evacuation/rescue route highlighted
  - Layer toggles and tooltips
"""

import math
import random
import folium
from folium.plugins import FeatureGroupSubGroup
from typing import Optional


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def _damage_colour(dmg: float) -> str:
    """Return hex colour for a damage fraction 0–1."""
    if dmg >= 0.7:   return "#e53e3e"   # red
    if dmg >= 0.4:   return "#dd6b20"   # orange
    if dmg >= 0.15:  return "#d69e2e"   # yellow
    return "#38a169"                    # green


def generate_map(graph, cfg: dict, disaster_summary: dict, world: dict, output_path: str):
    """
    Build and save the interactive Folium map.

    Parameters
    ----------
    graph           HybridTokyoGraph instance (post-disaster)
    cfg             dict with lat, lon, type, severity, radius, agent_radius
    disaster_summary dict from graph.apply_disaster()
    world           dict from build_world() — contains route_data, snap
    output_path     file path for the .html output
    """
    lat, lon = cfg["lat"], cfg["lon"]

    m = folium.Map(
        location=[lat, lon],
        zoom_start=14,
        tiles="CartoDB positron",
    )

    # Master layer control groups
    road_group    = folium.FeatureGroup(name="Roads", show=True)
    building_group = folium.FeatureGroup(name="Buildings", show=True)
    hospital_group = folium.FeatureGroup(name="Hospitals", show=True)
    firestation_group = folium.FeatureGroup(name="Fire Stations", show=True)
    route_group   = folium.FeatureGroup(name="Evacuation Route", show=True)
    disaster_group = folium.FeatureGroup(name="Disaster Zone", show=True)

    # -----------------------------------------------------------------------
    # 1. Disaster zone (epicentre + radius circle)
    # -----------------------------------------------------------------------
    folium.CircleMarker(
        location=[lat, lon],
        radius=10,
        color="#e53e3e",
        fill=True,
        fill_color="#e53e3e",
        fill_opacity=0.9,
        tooltip=folium.Tooltip(
            f"<b>Epicentre</b><br>"
            f"Type: {cfg['type'].upper()}<br>"
            f"Severity: {cfg['severity']:.0%}<br>"
            f"Radius: {cfg['radius']}m<br>"
            f"Population at risk: {disaster_summary['population_at_risk']:,}"
        ),
    ).add_to(disaster_group)

    folium.Circle(
        location=[lat, lon],
        radius=cfg["radius"],
        color="#e53e3e",
        weight=2,
        fill=True,
        fill_color="#e53e3e",
        fill_opacity=0.08,
        tooltip="Disaster impact zone",
    ).add_to(disaster_group)

    # -----------------------------------------------------------------------
    # 2. Road network — sample edges near epicentre for performance
    #    Color: red=blocked, yellow=damaged, green=safe
    # -----------------------------------------------------------------------
    MAX_ROAD_EDGES = 3000
    deg = (cfg["radius"] * 2.5) / 111_000
    seen_edges: set = set()
    edge_count = 0

    for nid, neighbours in graph._adj.items():
        node = graph._nodes.get(nid)
        if not node: continue
        if abs(node["lat"] - lat) > deg or abs(node["lon"] - lon) > deg:
            continue
        for nb_id, edge in neighbours:
            if nid > nb_id: continue   # draw once
            key = (nid, nb_id)
            if key in seen_edges: continue
            seen_edges.add(key)
            if edge_count >= MAX_ROAD_EDGES: break

            nb = graph._nodes.get(nb_id)
            if not nb: continue

            blocked  = not graph.is_edge_passable(nid, nb_id)
            dmg_u    = graph.get_node_damage(nid)
            dmg_v    = graph.get_node_damage(nb_id)
            avg_dmg  = (dmg_u + dmg_v) / 2

            if blocked:
                color, weight, opacity = "#e53e3e", 3, 0.9
                status = "BLOCKED"
            elif avg_dmg > 0.3:
                color, weight, opacity = "#d69e2e", 2, 0.8
                status = f"DAMAGED ({avg_dmg:.0%})"
            else:
                color, weight, opacity = "#48bb78", 1.5, 0.5
                status = "Safe"

            folium.PolyLine(
                locations=[[node["lat"], node["lon"]], [nb["lat"], nb["lon"]]],
                color=color,
                weight=weight,
                opacity=opacity,
                tooltip=f"Road: {status} | type: {edge.get('road_type','?')} | {edge.get('length',0):.0f}m",
            ).add_to(road_group)
            edge_count += 1
        if edge_count >= MAX_ROAD_EDGES:
            break

    # -----------------------------------------------------------------------
    # 3. Buildings — coloured by damage
    # -----------------------------------------------------------------------
    snap = world.get("snap", {})
    buildings_near = snap.get("buildings", [])

    # Also show all buildings in the disaster zone (up to 300)
    zone_buildings = graph.get_nodes_in_radius(lat, lon, cfg["radius"], "building")
    all_buildings  = list({b["id"]: b for b in buildings_near + zone_buildings}.values())
    random.shuffle(all_buildings)

    for b in all_buildings[:300]:
        dmg = graph.get_node_damage(b["id"])
        pop = b.get("population", 0)
        folium.CircleMarker(
            location=[b["lat"], b["lon"]],
            radius=max(3, min(8, pop // 150)),
            color=_damage_colour(dmg),
            fill=True,
            fill_color=_damage_colour(dmg),
            fill_opacity=0.75,
            tooltip=folium.Tooltip(
                f"<b>Building {b['id']}</b><br>"
                f"Population: {pop:,}<br>"
                f"Damage: {dmg:.0%}"
            ),
        ).add_to(building_group)

    # -----------------------------------------------------------------------
    # 4. Hospitals
    # -----------------------------------------------------------------------
    hospitals = graph.get_nodes_in_radius(lat, lon, cfg["radius"] * 2, "hospital")
    for h in hospitals[:60]:
        dmg = graph.get_node_damage(h["id"])
        status = "COMPROMISED" if dmg > 0.5 else "Operational"
        folium.Marker(
            location=[h["lat"], h["lon"]],
            icon=folium.Icon(color="red" if dmg > 0.5 else "blue",
                             icon="plus-sign", prefix="glyphicon"),
            tooltip=folium.Tooltip(
                f"<b>Hospital {h['id']}</b><br>"
                f"Status: {status}<br>"
                f"Damage: {dmg:.0%}"
            ),
        ).add_to(hospital_group)

    # -----------------------------------------------------------------------
    # 5. Fire stations
    # -----------------------------------------------------------------------
    fire_stations = graph.get_nodes_in_radius(lat, lon, cfg["radius"] * 2, "fire_station")
    for fs in fire_stations[:60]:
        dmg = graph.get_node_damage(fs["id"])
        status = "COMPROMISED" if dmg > 0.5 else "Active"
        folium.Marker(
            location=[fs["lat"], fs["lon"]],
            icon=folium.Icon(color="orange" if dmg > 0.5 else "darkred",
                             icon="fire", prefix="fa"),
            tooltip=folium.Tooltip(
                f"<b>Fire Station {fs['id']}</b><br>"
                f"Status: {status}<br>"
                f"Damage: {dmg:.0%}"
            ),
        ).add_to(firestation_group)

    # -----------------------------------------------------------------------
    # 6. Evacuation / rescue route
    # -----------------------------------------------------------------------
    route_data = world.get("route_data")
    if route_data and route_data.get("path"):
        path = route_data["path"]
        coords = []
        for nid in path:
            n = graph._nodes.get(nid)
            if n:
                coords.append([n["lat"], n["lon"]])

        if len(coords) >= 2:
            folium.PolyLine(
                locations=coords,
                color="#6b46c1",
                weight=5,
                opacity=0.9,
                tooltip=folium.Tooltip(
                    f"<b>Evacuation Route (A*)</b><br>"
                    f"Distance: {route_data['dist_km']:.2f} km<br>"
                    f"Est. time: {route_data['tt_min']:.1f} min<br>"
                    f"Hops: {len(path)} nodes<br>"
                    f"FS {route_data['src_id']} → Hospital {route_data['dst_id']}"
                ),
            ).add_to(route_group)

            # Start marker (fire station)
            src = route_data["src_node"]
            folium.Marker(
                location=[src["lat"], src["lon"]],
                icon=folium.Icon(color="purple", icon="play", prefix="fa"),
                tooltip=f"Route start: Fire Station {route_data['src_id']}",
            ).add_to(route_group)

            # End marker (hospital)
            dst = route_data["dst_node"]
            folium.Marker(
                location=[dst["lat"], dst["lon"]],
                icon=folium.Icon(color="green", icon="flag", prefix="fa"),
                tooltip=f"Route end: Hospital {route_data['dst_id']} "
                        f"({route_data['dist_km']:.2f} km, {route_data['tt_min']:.1f} min)",
            ).add_to(route_group)
    else:
        # Show staging zones if no route
        staging = graph.find_staging_zones((lat, lon), cfg["radius"])
        for z in staging:
            folium.Marker(
                location=[z["lat"], z["lon"]],
                icon=folium.Icon(color="beige", icon="helicopter", prefix="fa"),
                tooltip=f"Staging zone: FS {z['id']} | {z['dist_to_epicentre_m']}m from epicentre",
            ).add_to(route_group)

    # -----------------------------------------------------------------------
    # 7. Legend (HTML overlay)
    # -----------------------------------------------------------------------
    legend_html = """
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 1000;
        background: white; padding: 14px 18px; border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25); font-size: 13px;
        font-family: sans-serif; line-height: 1.7;
    ">
    <b>Legend</b><br>
    <span style="color:#e53e3e">&#9632;</span> Blocked road / Epicentre<br>
    <span style="color:#d69e2e">&#9632;</span> Damaged road / Building<br>
    <span style="color:#48bb78">&#9632;</span> Safe road / Building<br>
    <span style="color:#6b46c1">&#9632;</span> Evacuation route (A*)<br>
    <img src="https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-blue.png"
         style="height:16px;vertical-align:middle"> Hospital<br>
    <span style="color:#c05621">&#9650;</span> Fire station<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add all groups to map
    for group in [road_group, building_group, hospital_group,
                  firestation_group, route_group, disaster_group]:
        group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    m.save(output_path)
    return output_path
