from __future__ import annotations

import folium

from utils.domain import DUBAI_CENTER, ZONE_NAMES, ZONES, get_nearest_zone as domain_get_nearest_zone


def get_nearest_zone(lat: float, lon: float) -> str:
    return domain_get_nearest_zone(lat, lon)


def build_picker_map(
    pickup_point: tuple[float, float] | None,
    dropoff_point: tuple[float, float] | None,
    route_geometry: list[tuple[float, float]] | None = None,
):
    map_object = folium.Map(location=DUBAI_CENTER, zoom_start=11, tiles="CartoDB positron", control_scale=True)

    for zone_name in ZONE_NAMES:
        zone = ZONES[zone_name]
        folium.CircleMarker(
            location=(zone["lat"], zone["lon"]),
            radius=4,
            color="#134e4a",
            weight=1,
            fill=True,
            fill_opacity=0.65,
            fill_color="#0d9488",
            tooltip=zone_name,
        ).add_to(map_object)

    if route_geometry:
        folium.PolyLine(route_geometry, color="#f59e0b", weight=5, opacity=0.85).add_to(map_object)

    if pickup_point:
        folium.CircleMarker(
            location=pickup_point,
            radius=9,
            color="#166534",
            weight=2,
            fill=True,
            fill_color="#22c55e",
            fill_opacity=0.95,
            tooltip=f"Pickup | {get_nearest_zone(*pickup_point)}",
        ).add_to(map_object)
    if dropoff_point:
        folium.CircleMarker(
            location=dropoff_point,
            radius=9,
            color="#991b1b",
            weight=2,
            fill=True,
            fill_color="#ef4444",
            fill_opacity=0.95,
            tooltip=f"Dropoff | {get_nearest_zone(*dropoff_point)}",
        ).add_to(map_object)

    return map_object