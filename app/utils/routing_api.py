from __future__ import annotations

from datetime import datetime

import requests

from utils.config import get_config_value
from utils.domain import bearing_deg, build_fallback_route_context, classify_traffic, get_nearest_zone, haversine_km, stable_rng


def _fetch_osrm_route(pickup_lat: float, pickup_lon: float, dropoff_lat: float, dropoff_lon: float):
    response = requests.get(
        f"https://router.project-osrm.org/route/v1/driving/{pickup_lon},{pickup_lat};{dropoff_lon},{dropoff_lat}",
        params={"overview": "full", "geometries": "geojson"},
        timeout=8,
    )
    response.raise_for_status()
    payload = response.json()
    routes = payload.get("routes") or []
    if not routes:
        raise ValueError("No route returned from OSRM")
    route = routes[0]
    geometry = [(float(lat), float(lon)) for lon, lat in route["geometry"]["coordinates"]]
    return {
        "distance_km": float(route["distance"]) / 1000.0,
        "duration_min": float(route["duration"]) / 60.0,
        "route_geometry": geometry,
        "route_source": "OSRM",
    }


def _fetch_tomtom_traffic(lat: float, lon: float):
    api_key = get_config_value("TOMTOM_API_KEY")
    if not api_key:
        return None
    response = requests.get(
        "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json",
        params={"key": api_key, "point": f"{lat},{lon}"},
        timeout=8,
    )
    response.raise_for_status()
    payload = response.json().get("flowSegmentData", {})
    current_speed = float(payload.get("currentSpeed", 0.0) or 0.0)
    free_flow_speed = float(payload.get("freeFlowSpeed", 0.0) or 0.0)
    if current_speed <= 0 or free_flow_speed <= 0:
        raise ValueError("Incomplete TomTom flow payload")
    traffic_index = max(0.75, min(2.30, free_flow_speed / max(current_speed, 5.0)))
    return {
        "traffic_index": round(traffic_index, 3),
        "traffic_source": "TomTom Flow",
        "traffic_condition": classify_traffic(traffic_index),
    }


def _synthetic_traffic(ride_dt: datetime, direct_distance_km: float, pickup_lat: float, pickup_lon: float, dropoff_lat: float, dropoff_lon: float):
    rng = stable_rng("synthetic-traffic", round(pickup_lat, 4), round(pickup_lon, 4), round(dropoff_lat, 4), round(dropoff_lon, 4), ride_dt.isoformat())
    is_peak = ride_dt.hour >= 16 if ride_dt.weekday() == 4 else ((8 <= ride_dt.hour < 10) or (16 <= ride_dt.hour < 20))
    is_weekend = ride_dt.weekday() in (4, 5)
    traffic_index = min(
        2.20,
        max(
            0.70,
            0.84
            + 0.34 * float(is_peak)
            + 0.10 * float(is_weekend)
            + 0.06 * float(direct_distance_km > 18)
            + rng.normal(0, 0.04),
        ),
    )
    return {
        "traffic_index": round(float(traffic_index), 3),
        "traffic_source": "Synthetic traffic model",
        "traffic_condition": classify_traffic(float(traffic_index)),
    }


def get_route_context(
    pickup_lat: float,
    pickup_lon: float,
    dropoff_lat: float,
    dropoff_lon: float,
    ride_dt: datetime,
    prefer_live_traffic: bool = True,
):
    fallback = build_fallback_route_context(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, ride_dt)
    direct_distance_km = haversine_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

    try:
        route = _fetch_osrm_route(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    except (requests.RequestException, ValueError):
        return fallback

    traffic = None
    midpoint_lat = (pickup_lat + dropoff_lat) / 2.0
    midpoint_lon = (pickup_lon + dropoff_lon) / 2.0
    if prefer_live_traffic and get_config_value("TOMTOM_API_KEY") and ride_dt.date() == datetime.now().date():
        try:
            traffic = _fetch_tomtom_traffic(midpoint_lat, midpoint_lon)
        except (requests.RequestException, ValueError):
            traffic = None
    if traffic is None:
        traffic = _synthetic_traffic(ride_dt, direct_distance_km, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

    duration_min = route["duration_min"] * float(traffic["traffic_index"])
    return {
        "pickup_zone": get_nearest_zone(pickup_lat, pickup_lon),
        "dropoff_zone": get_nearest_zone(dropoff_lat, dropoff_lon),
        "distance_km": round(route["distance_km"], 2),
        "direct_distance_km": round(direct_distance_km, 2),
        "efficiency_ratio": round(route["distance_km"] / max(direct_distance_km, 0.5), 3),
        "duration_min": round(duration_min, 1),
        "bearing_deg": round(bearing_deg(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon), 2),
        "traffic_index": traffic["traffic_index"],
        "traffic_source": traffic["traffic_source"],
        "traffic_condition": traffic["traffic_condition"],
        "route_source": route["route_source"],
        "route_geometry": route["route_geometry"],
    }