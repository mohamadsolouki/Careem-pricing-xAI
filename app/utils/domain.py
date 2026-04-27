from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ── Project-root imports ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
for _p in (PROJECT_ROOT, MODELS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from feature_engineering import prepare_inference_frame
from zone_config import (
    ZONE_META,
    ZONE_NAMES,
    SALIK as _SALIK_MAP,
    get_zone_for_point,
    get_salik,
)

# Backward-compat alias (geo_utils.py and callers reference ZONES / ZONE_NAMES)
ZONES = ZONE_META
SALIK = _SALIK_MAP

# ── OSRM zone-pair distance matrix ───────────────────────────────────────────
_OSRM_DIST: dict[str, dict] = {}
_dist_json = PROJECT_ROOT / "data" / "processed" / "zone_distances.json"
if _dist_json.exists():
    with _dist_json.open() as _fh:
        _OSRM_DIST = json.load(_fh)

# ── Constants ─────────────────────────────────────────────────────────────────
DUBAI_CENTER = (25.2048, 55.2708)
DEFAULT_PICKUP_POINT = (25.2048, 55.2708)
DEFAULT_DROPOFF_POINT = (25.0815, 55.1403)
PRODUCT_NAMES = [
    "Comfort", "Executive", "Hala Taxi", "Eco Friendly", "Electric",
    "Kids", "Hala Kids", "Premier", "MAX", "Hala MAX",
]
PAYMENT_METHODS = ["Credit Card", "Cash", "Careem Pay", "Careem Plus"]

RAMADAN_START = pd.Timestamp("2025-03-01")
RAMADAN_END   = pd.Timestamp("2025-03-29")
UAE_HOLIDAYS = {
    "2025-01-01", "2025-03-30", "2025-03-31", "2025-04-01", "2025-04-02",
    "2025-04-03", "2025-06-05", "2025-06-06", "2025-06-07", "2025-06-08",
    "2025-06-09", "2025-06-26", "2025-09-04", "2025-12-02", "2025-12-03",
}

# Events calendar with venue zones (None = city-wide event)
EVENTS = [
    {"name": "Dubai Shopping Festival", "type": "Shopping Festival",
     "start": "2025-01-03", "end": "2025-02-01", "dmult": 1.35, "venue_zone": "Deira"},
    {"name": "Dubai Food Festival",     "type": "Food Festival",
     "start": "2025-02-20", "end": "2025-03-08", "dmult": 1.20, "venue_zone": "Jumeirah"},
    {"name": "Art Dubai",               "type": "Art/Culture",
     "start": "2025-03-18", "end": "2025-03-23", "dmult": 1.25, "venue_zone": "DIFC"},
    {"name": "Dubai World Cup",         "type": "Sports Event",
     "start": "2025-03-29", "end": "2025-03-29", "dmult": 1.55, "venue_zone": "Mirdif"},
    {"name": "Eid Al Fitr",             "type": "Religious Holiday",
     "start": "2025-03-30", "end": "2025-04-02", "dmult": 1.30, "venue_zone": None},
    {"name": "Formula 1 Weekend",       "type": "Sports Event",
     "start": "2025-04-04", "end": "2025-04-06", "dmult": 1.50, "venue_zone": "Yas Island"},
    {"name": "Eid Al Adha",             "type": "Religious Holiday",
     "start": "2025-06-05", "end": "2025-06-09", "dmult": 1.25, "venue_zone": None},
    {"name": "GITEX Global",            "type": "Tech Conference",
     "start": "2025-10-13", "end": "2025-10-17", "dmult": 1.45, "venue_zone": "DIFC"},
    {"name": "Diwali",                  "type": "Cultural Event",
     "start": "2025-10-20", "end": "2025-10-21", "dmult": 1.20, "venue_zone": "Bur Dubai"},
    {"name": "Dubai Airshow",           "type": "Trade Show",
     "start": "2025-11-17", "end": "2025-11-21", "dmult": 1.40, "venue_zone": "Dubai South"},
    {"name": "UAE National Day",        "type": "National Holiday",
     "start": "2025-12-02", "end": "2025-12-03", "dmult": 1.35, "venue_zone": None},
    {"name": "NYE Burj Khalifa",        "type": "New Year Event",
     "start": "2025-12-31", "end": "2025-12-31", "dmult": 2.20, "venue_zone": "Downtown"},
]

WEATHER_PROFILES = {
    1:  {"temp": (20, 25), "hum": (60, 75), "rain_p": 0.045, "storm_p": 0.010},
    2:  {"temp": (21, 27), "hum": (55, 72), "rain_p": 0.035, "storm_p": 0.008},
    3:  {"temp": (24, 31), "hum": (50, 68), "rain_p": 0.030, "storm_p": 0.012},
    4:  {"temp": (28, 35), "hum": (40, 60), "rain_p": 0.020, "storm_p": 0.018},
    5:  {"temp": (33, 39), "hum": (40, 58), "rain_p": 0.005, "storm_p": 0.020},
    6:  {"temp": (35, 42), "hum": (50, 70), "rain_p": 0.002, "storm_p": 0.025},
    7:  {"temp": (36, 43), "hum": (55, 78), "rain_p": 0.002, "storm_p": 0.020},
    8:  {"temp": (36, 42), "hum": (55, 80), "rain_p": 0.002, "storm_p": 0.015},
    9:  {"temp": (32, 38), "hum": (55, 75), "rain_p": 0.005, "storm_p": 0.012},
    10: {"temp": (28, 34), "hum": (50, 68), "rain_p": 0.015, "storm_p": 0.010},
    11: {"temp": (23, 30), "hum": (55, 72), "rain_p": 0.030, "storm_p": 0.008},
    12: {"temp": (19, 25), "hum": (55, 72), "rain_p": 0.040, "storm_p": 0.008},
}

PRODUCT_SPECS = {
    "Comfort":     {"base_day": 5.00, "base_night": 5.50, "per_km": 2.50, "per_min": 0.40, "min_fare": 15.0, "book_peak": 5.00, "book_offpeak": 3.00, "book_night": 2.50, "is_hala": False},
    "Executive":   {"base_day": 5.00, "base_night": 5.50, "per_km": 3.20, "per_min": 0.55, "min_fare": 18.0, "book_peak": 6.00, "book_offpeak": 4.00, "book_night": 3.50, "is_hala": False},
    "Hala Taxi":   {"base_day": 5.00, "base_night": 5.50, "per_km": 2.20, "per_min": 0.50, "min_fare": 13.0, "book_peak": 7.50, "book_offpeak": 4.00, "book_night": 4.00, "is_hala": True},
    "Eco Friendly":{"base_day": 5.00, "base_night": 5.50, "per_km": 2.60, "per_min": 0.42, "min_fare": 15.0, "book_peak": 5.00, "book_offpeak": 3.50, "book_night": 3.00, "is_hala": False},
    "Electric":    {"base_day": 5.00, "base_night": 5.50, "per_km": 2.80, "per_min": 0.45, "min_fare": 16.0, "book_peak": 5.50, "book_offpeak": 4.00, "book_night": 3.50, "is_hala": False},
    "Kids":        {"base_day": 5.00, "base_night": 5.50, "per_km": 2.80, "per_min": 0.45, "min_fare": 18.0, "book_peak": 6.00, "book_offpeak": 4.50, "book_night": 3.50, "is_hala": False},
    "Hala Kids":   {"base_day": 5.00, "base_night": 5.50, "per_km": 2.30, "per_min": 0.50, "min_fare": 15.0, "book_peak": 7.50, "book_offpeak": 4.50, "book_night": 4.00, "is_hala": True},
    "Premier":     {"base_day": 5.00, "base_night": 5.50, "per_km": 4.50, "per_min": 0.80, "min_fare": 30.0, "book_peak": 8.00, "book_offpeak": 6.00, "book_night": 5.00, "is_hala": False},
    "MAX":         {"base_day": 5.00, "base_night": 5.50, "per_km": 3.80, "per_min": 0.65, "min_fare": 25.0, "book_peak": 7.00, "book_offpeak": 5.00, "book_night": 4.50, "is_hala": False},
    "Hala MAX":    {"base_day": 5.00, "base_night": 5.50, "per_km": 2.80, "per_min": 0.60, "min_fare": 20.0, "book_peak": 8.00, "book_offpeak": 5.50, "book_night": 5.00, "is_hala": True},
}


# ── Utility functions ─────────────────────────────────────────────────────────

def stable_rng(*parts: object) -> np.random.Generator:
    digest = hashlib.sha256("||".join(map(str, parts)).encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16) % (2 ** 32)
    return np.random.default_rng(seed)


def normalize_calendar_date(ride_dt: datetime | pd.Timestamp) -> pd.Timestamp:
    """Return ride_dt as a pandas Timestamp unchanged (no year coercion)."""
    return pd.Timestamp(ride_dt)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi     = np.deg2rad(lat2 - lat1)
    dlambda  = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return float(radius_km * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dlambda = np.deg2rad(lon2 - lon1)
    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    return float((np.rad2deg(np.arctan2(y, x)) + 360.0) % 360.0)


def get_nearest_zone(lat: float, lon: float) -> str:
    """Polygon-based zone detection with nearest-centroid fallback."""
    return get_zone_for_point(lat, lon)


def classify_traffic(traffic_index: float) -> str:
    if traffic_index >= 1.65:
        return "Severe"
    if traffic_index >= 1.30:
        return "Heavy"
    if traffic_index >= 0.95:
        return "Moderate"
    return "Light"


def event_multiplier_for_zone(ev: dict, zone: str) -> float:
    """Return location-aware event demand multiplier for the given zone."""
    venue = ev.get("venue_zone")
    if venue is None:
        return ev["dmult"]          # city-wide events (Eid, National Day, etc.)
    if venue not in ZONE_META:
        return 1.0 + 0.30 * (ev["dmult"] - 1.0)   # off-map venue → mild city effect

    dist_km = haversine_km(
        ZONE_META[venue]["lat"], ZONE_META[venue]["lon"],
        ZONE_META.get(zone, ZONE_META["Downtown"])["lat"],
        ZONE_META.get(zone, ZONE_META["Downtown"])["lon"],
    )
    if dist_km < 3:
        decay = 1.00
    elif dist_km < 8:
        decay = 0.75
    elif dist_km < 18:
        decay = 0.40
    else:
        decay = 0.10
    return 1.0 + decay * (ev["dmult"] - 1.0)


# ── Domain logic ──────────────────────────────────────────────────────────────

def get_event_context(
    ride_dt: datetime | pd.Timestamp,
    pickup_zone: str | None = None,
) -> dict[str, object]:
    dt = pd.Timestamp(ride_dt).normalize()
    for event in EVENTS:
        start = pd.Timestamp(event["start"])
        end   = pd.Timestamp(event["end"])
        if start <= dt <= end:
            dmult = (
                event_multiplier_for_zone(event, pickup_zone)
                if pickup_zone else event["dmult"]
            )
            return {
                "active_event": event["name"],
                "event_type":   event["type"],
                "event_demand_multiplier": float(dmult),
            }
    return {"active_event": "None", "event_type": "None", "event_demand_multiplier": 1.0}


def get_time_context(ride_dt: datetime | pd.Timestamp) -> dict[str, object]:
    actual_dt = pd.Timestamp(ride_dt)
    hour  = int(actual_dt.hour)
    dow   = int(actual_dt.dayofweek)
    # UAE work week Mon–Fri (0–4); weekend = Saturday (5) + Sunday (6)
    is_weekend   = dow in (5, 6)
    is_peak_hour = (8 <= hour < 10) or (16 <= hour < 20)
    is_late_night = hour >= 22 or hour < 6
    is_offpeak    = not is_peak_hour and not is_late_night
    cal_str       = actual_dt.strftime("%Y-%m-%d")
    is_ramadan     = bool(RAMADAN_START <= actual_dt.normalize() <= RAMADAN_END)
    is_suhoor      = bool(is_ramadan and 1 <= hour <= 3)
    is_iftar       = bool(is_ramadan and hour == 17)
    is_public_holiday = cal_str in UAE_HOLIDAYS
    return {
        "timestamp":          actual_dt,
        "date":               actual_dt.date().isoformat(),
        "hour":               hour,
        "minute":             int(actual_dt.minute),
        "day_of_week":        dow,
        "day_name":           actual_dt.day_name(),
        "week_of_year":       int(actual_dt.isocalendar().week),
        "month":              int(actual_dt.month),
        "month_name":         actual_dt.month_name(),
        "quarter":            int(actual_dt.quarter),
        "is_weekend":         is_weekend,
        "is_peak_hour":       is_peak_hour,
        "is_late_night":      is_late_night,
        "is_offpeak":         is_offpeak,
        "is_ramadan":         is_ramadan,
        "is_uae_public_holiday": is_public_holiday,
        "is_suhoor_window":   is_suhoor,
        "is_iftar_window":    is_iftar,
    }


def get_distance_km(
    pickup_lat: float, pickup_lon: float,
    dropoff_lat: float, dropoff_lon: float,
) -> float:
    """Estimate road distance using OSRM zone-pair matrix, with haversine fallback."""
    pu_zone = get_zone_for_point(pickup_lat, pickup_lon)
    do_zone = get_zone_for_point(dropoff_lat, dropoff_lon)
    rng = stable_rng(
        "distance",
        round(pickup_lat, 5), round(pickup_lon, 5),
        round(dropoff_lat, 5), round(dropoff_lon, 5),
    )
    direct_dist = haversine_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

    if pu_zone == do_zone:
        return round(float(max(direct_dist * rng.uniform(1.10, 1.45), rng.uniform(1.8, 4.2))), 2)

    key     = f"{pu_zone}|{do_zone}"
    rev_key = f"{do_zone}|{pu_zone}"
    if key in _OSRM_DIST:
        base = _OSRM_DIST[key]["distance_km"]
    elif rev_key in _OSRM_DIST:
        base = _OSRM_DIST[rev_key]["distance_km"]
    else:
        base = max(direct_dist * 1.20, 15.0)

    return round(float(max(base * rng.uniform(0.97, 1.03), direct_dist * rng.uniform(1.08, 1.35))), 2)


def get_salik_gates(pickup_zone: str, dropoff_zone: str) -> int:
    return get_salik(pickup_zone, dropoff_zone)


def build_fallback_route_context(
    pickup_lat: float,
    pickup_lon: float,
    dropoff_lat: float,
    dropoff_lon: float,
    ride_dt: datetime | pd.Timestamp,
    weather_dmult: float = 1.0,
) -> dict[str, object]:
    pickup_zone  = get_zone_for_point(pickup_lat, pickup_lon)
    dropoff_zone = get_zone_for_point(dropoff_lat, dropoff_lon)
    time_ctx     = get_time_context(ride_dt)
    ev_ctx       = get_event_context(ride_dt, pickup_zone)
    event_dmult  = float(ev_ctx["event_demand_multiplier"])

    rng = stable_rng(
        "route-context",
        round(pickup_lat, 5), round(pickup_lon, 5),
        round(dropoff_lat, 5), round(dropoff_lon, 5),
        time_ctx["timestamp"].isoformat(),
    )

    direct_dist_km  = haversine_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    distance_km     = get_distance_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    efficiency_ratio = distance_km / max(direct_dist_km, 0.5)
    is_airport_ride  = pickup_zone == "DXB Airport" or dropoff_zone == "DXB Airport"

    # ── Unified traffic formula (matches generate_dataset.py) ────────────────
    traffic_index = float(np.clip(
        0.82
        + 0.32 * float(time_ctx["is_peak_hour"])
        + 0.10 * float(time_ctx["is_weekend"])
        + 0.14 * (event_dmult - 1.0)
        + 0.10 * (weather_dmult - 1.0)
        + 0.08 * float(is_airport_ride)
        + 0.06 * float(np.clip(efficiency_ratio - 1.0, 0.0, 1.5))
        + rng.normal(0, 0.04),
        0.68, 2.20,
    ))

    free_flow_speed = (
        rng.uniform(38, 52) if time_ctx["is_peak_hour"]
        else (rng.uniform(58, 72) if time_ctx["is_late_night"] else rng.uniform(44, 58))
    )
    avg_speed_kmh = float(np.clip(free_flow_speed / traffic_index, 12.0, 78.0))
    duration_min  = float((distance_km / max(avg_speed_kmh, 1.0)) * 60)

    return {
        "pickup_zone":          pickup_zone,
        "dropoff_zone":         dropoff_zone,
        "distance_km":          round(distance_km, 2),
        "direct_distance_km":   round(direct_dist_km, 2),
        "efficiency_ratio":     round(efficiency_ratio, 3),
        "duration_min":         round(duration_min, 1),
        "bearing_deg":          round(bearing_deg(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon), 2),
        "traffic_index":        round(traffic_index, 3),
        "traffic_source":       "Synthetic traffic model",
        "traffic_condition":    classify_traffic(traffic_index),
        "route_source":         "Synthetic route model",
        "route_geometry":       [(pickup_lat, pickup_lon), (dropoff_lat, dropoff_lon)],
    }


def build_trip_record(
    pickup_lat: float,
    pickup_lon: float,
    dropoff_lat: float,
    dropoff_lon: float,
    product_type: str,
    ride_dt: datetime | pd.Timestamp,
    weather: dict[str, object],
    payment_method: str,
    route_context: dict[str, object] | None = None,
) -> dict[str, object]:
    time_ctx  = get_time_context(ride_dt)
    weather_dmult = float(weather.get("weather_demand_factor", 1.0))

    if route_context is None:
        route_context = build_fallback_route_context(
            pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
            ride_dt, weather_dmult=weather_dmult,
        )

    pickup_zone  = str(route_context.get("pickup_zone") or get_zone_for_point(pickup_lat, pickup_lon))
    dropoff_zone = str(route_context.get("dropoff_zone") or get_zone_for_point(dropoff_lat, dropoff_lon))

    # Gracefully handle unknown zones
    pickup_meta  = ZONE_META.get(pickup_zone,  ZONE_META["Downtown"])
    dropoff_meta = ZONE_META.get(dropoff_zone, ZONE_META["Downtown"])

    ev_ctx = get_event_context(ride_dt, pickup_zone)

    rng = stable_rng(
        "scenario",
        round(pickup_lat, 5), round(pickup_lon, 5),
        round(dropoff_lat, 5), round(dropoff_lon, 5),
        product_type,
        time_ctx["timestamp"].isoformat(),
        payment_method,
    )

    route_distance_km      = float(route_context["distance_km"])
    route_direct_dist_km   = float(route_context["direct_distance_km"])
    route_efficiency_ratio = float(route_context["efficiency_ratio"])
    route_bearing          = float(route_context["bearing_deg"])
    traffic_index          = float(route_context["traffic_index"])
    trip_duration_min      = float(route_context["duration_min"])

    salik_gates    = get_salik_gates(pickup_zone, dropoff_zone)
    salik_cost_aed = round(salik_gates * 4.0, 2)
    is_airport_ride  = pickup_zone == "DXB Airport" or dropoff_zone == "DXB Airport"
    is_intrazone_trip = pickup_zone == dropoff_zone
    is_hala_product   = bool(PRODUCT_SPECS[product_type]["is_hala"])
    is_careem_plus    = payment_method == "Careem Plus"

    # ── Demand index ─────────────────────────────────────────────────────────
    temporal_demand = 1.18 if time_ctx["is_peak_hour"] else (0.92 if time_ctx["is_late_night"] else 1.00)
    ramadan_demand  = (1.35 if time_ctx["is_iftar_window"]
                       else (1.15 if time_ctx["is_suhoor_window"]
                             else (0.96 if time_ctx["is_ramadan"] else 1.00)))
    weekend_demand  = 1.05 if time_ctx["is_weekend"] else 1.00
    demand_index = float(np.clip(
        pickup_meta["dmult"]
        * float(ev_ctx["event_demand_multiplier"])
        * weather_dmult
        * temporal_demand
        * ramadan_demand
        * weekend_demand,
        0.75 + 0.05 * max(traffic_index - 1.0, 0.0),
        3.00,
    ))

    captain_availability_score = float(np.clip(
        1.0 - 0.30 * (demand_index - 1.0) - 0.05 * max(traffic_index - 1.0, 0.0)
        + rng.normal(0, 0.06),
        0.15, 1.0,
    ))
    supply_pressure_index = float(np.clip(1.0 - captain_availability_score, 0.0, 1.0))

    wait_base    = 5.5 if is_airport_ride else (4.5 if time_ctx["is_peak_hour"] else 3.0)
    wait_time_min = float(np.clip(
        wait_base * (1.5 - captain_availability_score) * (0.95 + 0.20 * traffic_index)
        + rng.uniform(0.4, 2.2),
        1.0, 25.0,
    ))
    avg_speed_kmh = float(np.clip(
        route_distance_km / max(trip_duration_min / 60.0, 0.15),
        12.0, 78.0,
    ))

    # ── Pricing ──────────────────────────────────────────────────────────────
    product       = PRODUCT_SPECS[product_type]
    flagfall      = product["base_night"] if time_ctx["is_late_night"] else product["base_day"]
    booking_fee   = (
        product["book_peak"] if time_ctx["is_peak_hour"]
        else (product["book_night"] if time_ctx["is_late_night"] else product["book_offpeak"])
    )

    if is_hala_product:
        start_charge   = 25.0 if pickup_zone == "DXB Airport" else flagfall + booking_fee
        metered_fare   = start_charge + product["per_km"] * route_distance_km + wait_time_min * 0.50 + salik_cost_aed
        surge_mult     = 1.0
        final_price    = max(metered_fare, product["min_fare"])
    else:
        demand_supply_gap = float(np.clip(demand_index - captain_availability_score, 0.0, 1.5))
        surge_raw  = float(np.clip(1.0 + demand_supply_gap * 0.55, 1.0, 2.5))
        surge_mult = round(round(surge_raw * 4) / 4, 3)   # round to nearest 0.25×
        metered_fare = (
            flagfall + booking_fee
            + (product["per_km"] * route_distance_km + product["per_min"] * trip_duration_min) * surge_mult
            + salik_cost_aed
        )
        final_price = max(metered_fare, product["min_fare"])
        # Careem Plus 5% loyalty discount on private-hire fares
        if is_careem_plus:
            final_price = max(final_price * 0.95, product["min_fare"])

    return {
        "ride_id":        "SIMULATED-RIDE",
        "customer_id":    "SIM-CUSTOMER",
        "captain_id":     "SIM-CAPTAIN",
        "timestamp":      time_ctx["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
        "date":           time_ctx["date"],
        "hour":           time_ctx["hour"],
        "minute":         time_ctx["minute"],
        "day_of_week":    time_ctx["day_of_week"],
        "day_name":       time_ctx["day_name"],
        "week_of_year":   time_ctx["week_of_year"],
        "month":          time_ctx["month"],
        "month_name":     time_ctx["month_name"],
        "quarter":        time_ctx["quarter"],
        "is_weekend":     time_ctx["is_weekend"],
        "is_peak_hour":   time_ctx["is_peak_hour"],
        "is_late_night":  time_ctx["is_late_night"],
        "is_offpeak":     time_ctx["is_offpeak"],
        "is_ramadan":     time_ctx["is_ramadan"],
        "is_uae_public_holiday": time_ctx["is_uae_public_holiday"],
        "is_suhoor_window": time_ctx["is_suhoor_window"],
        "is_iftar_window":  time_ctx["is_iftar_window"],
        "active_event":    ev_ctx["active_event"],
        "event_type":      ev_ctx["event_type"],
        "event_demand_multiplier": round(float(ev_ctx["event_demand_multiplier"]), 3),
        "temperature_c":   round(float(weather["temperature_c"]), 1),
        "humidity_pct":    round(float(weather["humidity_pct"]), 1),
        "is_rain":         bool(weather["is_rain"]),
        "is_sandstorm":    bool(weather["is_sandstorm"]),
        "weather_demand_factor": round(float(weather_dmult), 3),
        "pickup_zone":     pickup_zone,
        "dropoff_zone":    dropoff_zone,
        "pickup_lat":      round(float(pickup_lat), 6),
        "pickup_lon":      round(float(pickup_lon), 6),
        "dropoff_lat":     round(float(dropoff_lat), 6),
        "dropoff_lon":     round(float(dropoff_lon), 6),
        "pickup_area_type":  pickup_meta["tier"],
        "dropoff_area_type": dropoff_meta["tier"],
        "is_airport_ride":   is_airport_ride,
        "is_intrazone_trip": is_intrazone_trip,
        "pickup_density_score":  round(float(pickup_meta["dmult"]), 3),
        "dropoff_density_score": round(float(dropoff_meta["dmult"]), 3),
        "route_direct_distance_km": round(route_direct_dist_km, 2),
        "route_distance_km":        round(route_distance_km, 2),
        "route_efficiency_ratio":   round(route_efficiency_ratio, 3),
        "route_bearing_deg":        round(route_bearing, 2),
        "salik_gates":     salik_gates,
        "salik_cost_aed":  round(salik_cost_aed, 2),
        "product_type":    product_type,
        "is_hala_product": is_hala_product,
        "payment_method":  payment_method,
        "is_careem_plus":  is_careem_plus,
        "demand_index":             round(demand_index, 3),
        "captain_availability_score": round(captain_availability_score, 3),
        "supply_pressure_index":    round(supply_pressure_index, 3),
        "traffic_index":            round(traffic_index, 3),
        "wait_time_min":            round(wait_time_min, 1),
        "trip_duration_min":        round(trip_duration_min, 1),
        "avg_speed_kmh":            round(avg_speed_kmh, 1),
        "surge_multiplier":         round(surge_mult, 3),
        "booking_fee_aed":          round(float(booking_fee), 2),
        "metered_fare_aed":         round(float(metered_fare), 2),
        "final_price_aed":          round(float(final_price), 2),
        "price_per_km_aed":         round(float(final_price / max(route_distance_km, 0.1)), 2),
        "booking_status":      "Completed",
        "cancellation_reason": "N/A",
        "captain_rating":      4.8,
        "customer_rating":     4.7,
        "eta_deviation_min":   0.0,
        "weather_source":      weather.get("source", "Unknown"),
        "weather_label":       weather.get("weather_label", "Clear"),
        "traffic_source":      route_context.get("traffic_source", "Synthetic traffic model"),
        "traffic_condition":   route_context.get("traffic_condition", classify_traffic(traffic_index)),
        "route_source":        route_context.get("route_source", "Synthetic route model"),
        "route_geometry":      route_context.get("route_geometry", [(pickup_lat, pickup_lon), (dropoff_lat, dropoff_lon)]),
    }


def build_inference_frame(record: dict[str, object], feature_columns: list[str]):
    frame = pd.DataFrame([record])
    X_row, _, _ = prepare_inference_frame(frame, feature_columns)
    return X_row, frame
