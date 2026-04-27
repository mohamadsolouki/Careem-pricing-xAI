from __future__ import annotations

from datetime import datetime

import requests

from utils.config import get_config_value
from utils.domain import (
    bearing_deg,
    build_fallback_route_context,
    classify_traffic,
    get_event_context,
    get_zone_for_point,
    haversine_km,
    stable_rng,
)


def _fetch_osrm_route(
    pickup_lat: float, pickup_lon: float,
    dropoff_lat: float, dropoff_lon: float,
):
    response = requests.get(
        f"https://router.project-osrm.org/route/v1/driving/"
        f"{pickup_lon},{pickup_lat};{dropoff_lon},{dropoff_lat}",
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
        "distance_km":    float(route["distance"]) / 1000.0,
        "duration_min":   float(route["duration"]) / 60.0,
        "route_geometry": geometry,
        "route_source":   "OSRM",
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
    current_speed   = float(payload.get("currentSpeed",  0.0) or 0.0)
    free_flow_speed = float(payload.get("freeFlowSpeed", 0.0) or 0.0)
    if current_speed <= 0 or free_flow_speed <= 0:
        raise ValueError("Incomplete TomTom flow payload")
    traffic_index = max(0.75, min(2.30, free_flow_speed / max(current_speed, 5.0)))
    return {
        "traffic_index":    round(traffic_index, 3),
        "traffic_source":   "TomTom Flow",
        "traffic_condition": classify_traffic(traffic_index),
    }


def _synthetic_traffic(
    ride_dt: datetime,
    pickup_lat: float, pickup_lon: float,
    dropoff_lat: float, dropoff_lon: float,
    efficiency_ratio: float = 1.20,
    weather_dmult: float = 1.0,
):
    """
    Unified synthetic traffic formula — matches generate_dataset.py exactly:
      0.82 + 0.32*peak + 0.10*weekend + 0.14*(event-1)
          + 0.10*(weather-1) + 0.08*airport + 0.06*efficiency_clip + N(0,0.04)
    """
    rng = stable_rng(
        "synthetic-traffic",
        round(pickup_lat, 4), round(pickup_lon, 4),
        round(dropoff_lat, 4), round(dropoff_lon, 4),
        ride_dt.isoformat(),
    )
    # UAE work week Mon–Fri (0–4); weekend = Saturday (5) + Sunday (6)
    is_peak    = (8 <= ride_dt.hour < 10) or (16 <= ride_dt.hour < 20)
    is_weekend = ride_dt.weekday() in (5, 6)
    pickup_zone    = get_zone_for_point(pickup_lat, pickup_lon)
    dropoff_zone   = get_zone_for_point(dropoff_lat, dropoff_lon)
    is_airport     = pickup_zone == "DXB Airport" or dropoff_zone == "DXB Airport"
    ev_ctx         = get_event_context(ride_dt, pickup_zone)
    event_dmult    = float(ev_ctx["event_demand_multiplier"])
    eff_clip       = float(min(max(efficiency_ratio - 1.0, 0.0), 1.5))

    raw = (
        0.82
        + 0.32 * float(is_peak)
        + 0.10 * float(is_weekend)
        + 0.14 * (event_dmult - 1.0)
        + 0.10 * (weather_dmult - 1.0)
        + 0.08 * float(is_airport)
        + 0.06 * eff_clip
        + float(rng.normal(0, 0.04))
    )
    traffic_index = float(max(0.68, min(2.20, raw)))
    return {
        "traffic_index":    round(traffic_index, 3),
        "traffic_source":   "Synthetic traffic model",
        "traffic_condition": classify_traffic(traffic_index),
    }


def get_route_context(
    pickup_lat: float,
    pickup_lon: float,
    dropoff_lat: float,
    dropoff_lon: float,
    ride_dt: datetime,
    prefer_live_traffic: bool = True,
    weather_dmult: float = 1.0,
):
    direct_dist_km = haversine_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

    try:
        route = _fetch_osrm_route(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    except (requests.RequestException, ValueError):
        return build_fallback_route_context(
            pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
            ride_dt, weather_dmult=weather_dmult,
        )

    traffic = None
    midpoint_lat = (pickup_lat + dropoff_lat) / 2.0
    midpoint_lon = (pickup_lon + dropoff_lon) / 2.0
    if (prefer_live_traffic
            and get_config_value("TOMTOM_API_KEY")
            and ride_dt.date() == datetime.now().date()):
        try:
            traffic = _fetch_tomtom_traffic(midpoint_lat, midpoint_lon)
        except (requests.RequestException, ValueError):
            traffic = None

    if traffic is None:
        efficiency_ratio = route["distance_km"] / max(direct_dist_km, 0.5)
        traffic = _synthetic_traffic(
            ride_dt,
            pickup_lat, pickup_lon,
            dropoff_lat, dropoff_lon,
            efficiency_ratio=efficiency_ratio,
            weather_dmult=weather_dmult,
        )

    duration_min = route["duration_min"] * float(traffic["traffic_index"])

    return {
        "pickup_zone":        get_zone_for_point(pickup_lat, pickup_lon),
        "dropoff_zone":       get_zone_for_point(dropoff_lat, dropoff_lon),
        "distance_km":        round(route["distance_km"], 2),
        "direct_distance_km": round(direct_dist_km, 2),
        "efficiency_ratio":   round(route["distance_km"] / max(direct_dist_km, 0.5), 3),
        "duration_min":       round(duration_min, 1),
        "bearing_deg":        round(bearing_deg(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon), 2),
        "traffic_index":      traffic["traffic_index"],
        "traffic_source":     traffic["traffic_source"],
        "traffic_condition":  traffic["traffic_condition"],
        "route_source":       route["route_source"],
        "route_geometry":     route["route_geometry"],
    }
