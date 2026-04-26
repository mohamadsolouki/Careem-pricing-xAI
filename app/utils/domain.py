from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
for path in (PROJECT_ROOT, MODELS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from feature_engineering import prepare_inference_frame


ZONES = {
    "Downtown": {"lat": 25.1972, "lon": 55.2744, "tier": "High", "dmult": 1.25},
    "Marina": {"lat": 25.0805, "lon": 55.1403, "tier": "High", "dmult": 1.20},
    "JBR": {"lat": 25.0774, "lon": 55.1302, "tier": "High", "dmult": 1.15},
    "DIFC": {"lat": 25.2118, "lon": 55.2797, "tier": "High", "dmult": 1.30},
    "Deira": {"lat": 25.2697, "lon": 55.3095, "tier": "Medium", "dmult": 1.05},
    "Bur Dubai": {"lat": 25.2532, "lon": 55.2956, "tier": "Medium", "dmult": 1.05},
    "Jumeirah": {"lat": 25.2048, "lon": 55.2434, "tier": "Medium", "dmult": 1.10},
    "Al Quoz": {"lat": 25.1521, "lon": 55.2270, "tier": "Low", "dmult": 0.90},
    "Business Bay": {"lat": 25.1867, "lon": 55.2622, "tier": "High", "dmult": 1.20},
    "Dubai Hills": {"lat": 25.1150, "lon": 55.2380, "tier": "Medium", "dmult": 1.00},
    "DXB Airport": {"lat": 25.2532, "lon": 55.3657, "tier": "High", "dmult": 1.05},
    "Sharjah": {"lat": 25.3463, "lon": 55.4209, "tier": "Low", "dmult": 0.95},
}

DIST_MATRIX = {
    ("Downtown", "Marina"): 24.0,
    ("Downtown", "JBR"): 26.0,
    ("Downtown", "DIFC"): 2.5,
    ("Downtown", "Deira"): 8.0,
    ("Downtown", "Bur Dubai"): 5.0,
    ("Downtown", "Jumeirah"): 9.0,
    ("Downtown", "Al Quoz"): 10.0,
    ("Downtown", "Business Bay"): 3.5,
    ("Downtown", "Dubai Hills"): 18.0,
    ("Downtown", "DXB Airport"): 14.0,
    ("Downtown", "Sharjah"): 25.0,
    ("Marina", "JBR"): 2.5,
    ("Marina", "DIFC"): 22.0,
    ("Marina", "Deira"): 34.0,
    ("Marina", "Bur Dubai"): 30.0,
    ("Marina", "Jumeirah"): 17.0,
    ("Marina", "Al Quoz"): 15.0,
    ("Marina", "Business Bay"): 23.0,
    ("Marina", "Dubai Hills"): 18.0,
    ("Marina", "DXB Airport"): 38.0,
    ("Marina", "Sharjah"): 50.0,
    ("JBR", "DIFC"): 24.0,
    ("JBR", "Deira"): 36.0,
    ("JBR", "Bur Dubai"): 32.0,
    ("JBR", "Jumeirah"): 17.0,
    ("JBR", "Al Quoz"): 16.0,
    ("JBR", "Business Bay"): 25.0,
    ("JBR", "Dubai Hills"): 20.0,
    ("JBR", "DXB Airport"): 40.0,
    ("JBR", "Sharjah"): 52.0,
    ("DIFC", "Deira"): 10.0,
    ("DIFC", "Bur Dubai"): 6.0,
    ("DIFC", "Jumeirah"): 10.0,
    ("DIFC", "Al Quoz"): 10.0,
    ("DIFC", "Business Bay"): 2.5,
    ("DIFC", "Dubai Hills"): 16.0,
    ("DIFC", "DXB Airport"): 15.0,
    ("DIFC", "Sharjah"): 27.0,
    ("Deira", "Bur Dubai"): 4.5,
    ("Deira", "Jumeirah"): 16.0,
    ("Deira", "Al Quoz"): 20.0,
    ("Deira", "Business Bay"): 12.0,
    ("Deira", "Dubai Hills"): 28.0,
    ("Deira", "DXB Airport"): 7.0,
    ("Deira", "Sharjah"): 18.0,
    ("Bur Dubai", "Jumeirah"): 13.0,
    ("Bur Dubai", "Al Quoz"): 14.0,
    ("Bur Dubai", "Business Bay"): 7.0,
    ("Bur Dubai", "Dubai Hills"): 23.0,
    ("Bur Dubai", "DXB Airport"): 12.0,
    ("Bur Dubai", "Sharjah"): 23.0,
    ("Jumeirah", "Al Quoz"): 8.0,
    ("Jumeirah", "Business Bay"): 9.0,
    ("Jumeirah", "Dubai Hills"): 16.0,
    ("Jumeirah", "DXB Airport"): 22.0,
    ("Jumeirah", "Sharjah"): 37.0,
    ("Al Quoz", "Business Bay"): 9.0,
    ("Al Quoz", "Dubai Hills"): 12.0,
    ("Al Quoz", "DXB Airport"): 22.0,
    ("Al Quoz", "Sharjah"): 38.0,
    ("Business Bay", "Dubai Hills"): 14.0,
    ("Business Bay", "DXB Airport"): 16.0,
    ("Business Bay", "Sharjah"): 28.0,
    ("Dubai Hills", "DXB Airport"): 28.0,
    ("Dubai Hills", "Sharjah"): 42.0,
    ("DXB Airport", "Sharjah"): 22.0,
}

SALIK = {
    ("Marina", "Downtown"): 2,
    ("Marina", "DIFC"): 2,
    ("Marina", "Business Bay"): 2,
    ("Marina", "Deira"): 3,
    ("Marina", "Bur Dubai"): 2,
    ("Marina", "DXB Airport"): 3,
    ("Marina", "Sharjah"): 3,
    ("JBR", "Downtown"): 2,
    ("JBR", "DIFC"): 2,
    ("JBR", "Business Bay"): 2,
    ("JBR", "Deira"): 3,
    ("JBR", "DXB Airport"): 3,
    ("JBR", "Sharjah"): 3,
    ("Al Quoz", "Deira"): 2,
    ("Al Quoz", "DXB Airport"): 2,
    ("Al Quoz", "Sharjah"): 2,
    ("Dubai Hills", "Deira"): 2,
    ("Dubai Hills", "DXB Airport"): 2,
    ("Dubai Hills", "Sharjah"): 3,
    ("DIFC", "DXB Airport"): 1,
    ("DIFC", "Sharjah"): 2,
    ("Downtown", "DXB Airport"): 1,
    ("Downtown", "Sharjah"): 2,
    ("Deira", "Marina"): 3,
    ("Deira", "JBR"): 3,
    ("Bur Dubai", "Marina"): 2,
    ("Bur Dubai", "JBR"): 2,
    ("DXB Airport", "Marina"): 3,
    ("DXB Airport", "JBR"): 3,
    ("DXB Airport", "Dubai Hills"): 2,
    ("Sharjah", "Marina"): 3,
    ("Sharjah", "JBR"): 3,
    ("Sharjah", "Dubai Hills"): 3,
}

EVENTS = [
    {"name": "Dubai Shopping Festival", "type": "Shopping Festival", "start": "2025-01-03", "end": "2025-02-01", "dmult": 1.35},
    {"name": "Dubai Food Festival", "type": "Food Festival", "start": "2025-02-20", "end": "2025-03-08", "dmult": 1.20},
    {"name": "Art Dubai", "type": "Art/Culture", "start": "2025-03-18", "end": "2025-03-23", "dmult": 1.25},
    {"name": "Dubai World Cup", "type": "Sports Event", "start": "2025-03-29", "end": "2025-03-29", "dmult": 1.55},
    {"name": "Eid Al Fitr", "type": "Religious Holiday", "start": "2025-03-30", "end": "2025-04-02", "dmult": 1.30},
    {"name": "Formula 1 Weekend", "type": "Sports Event", "start": "2025-04-04", "end": "2025-04-06", "dmult": 1.50},
    {"name": "Eid Al Adha", "type": "Religious Holiday", "start": "2025-06-05", "end": "2025-06-09", "dmult": 1.25},
    {"name": "GITEX Global", "type": "Tech Conference", "start": "2025-10-13", "end": "2025-10-17", "dmult": 1.45},
    {"name": "Diwali", "type": "Cultural Event", "start": "2025-10-20", "end": "2025-10-21", "dmult": 1.20},
    {"name": "Dubai Airshow", "type": "Trade Show", "start": "2025-11-17", "end": "2025-11-21", "dmult": 1.40},
    {"name": "UAE National Day", "type": "National Holiday", "start": "2025-12-02", "end": "2025-12-03", "dmult": 1.35},
    {"name": "NYE Burj Khalifa", "type": "New Year Event", "start": "2025-12-31", "end": "2025-12-31", "dmult": 2.20},
]

WEATHER_PROFILES = {
    1: {"temp": (20, 25), "hum": (60, 75), "rain_p": 0.045, "storm_p": 0.010},
    2: {"temp": (21, 27), "hum": (55, 72), "rain_p": 0.035, "storm_p": 0.008},
    3: {"temp": (24, 31), "hum": (50, 68), "rain_p": 0.030, "storm_p": 0.012},
    4: {"temp": (28, 35), "hum": (40, 60), "rain_p": 0.020, "storm_p": 0.018},
    5: {"temp": (33, 39), "hum": (40, 58), "rain_p": 0.005, "storm_p": 0.020},
    6: {"temp": (35, 42), "hum": (50, 70), "rain_p": 0.002, "storm_p": 0.025},
    7: {"temp": (36, 43), "hum": (55, 78), "rain_p": 0.002, "storm_p": 0.020},
    8: {"temp": (36, 42), "hum": (55, 80), "rain_p": 0.002, "storm_p": 0.015},
    9: {"temp": (32, 38), "hum": (55, 75), "rain_p": 0.005, "storm_p": 0.012},
    10: {"temp": (28, 34), "hum": (50, 68), "rain_p": 0.015, "storm_p": 0.010},
    11: {"temp": (23, 30), "hum": (55, 72), "rain_p": 0.030, "storm_p": 0.008},
    12: {"temp": (19, 25), "hum": (55, 72), "rain_p": 0.040, "storm_p": 0.008},
}

PRODUCT_SPECS = {
    "Comfort": {"base_day": 5.00, "base_night": 5.50, "per_km": 2.50, "per_min": 0.40, "min_fare": 15.0, "book_peak": 5.00, "book_offpeak": 3.00, "book_night": 2.50, "is_hala": False},
    "Executive": {"base_day": 5.00, "base_night": 5.50, "per_km": 3.20, "per_min": 0.55, "min_fare": 18.0, "book_peak": 6.00, "book_offpeak": 4.00, "book_night": 3.50, "is_hala": False},
    "Hala Taxi": {"base_day": 5.00, "base_night": 5.50, "per_km": 2.20, "per_min": 0.50, "min_fare": 13.0, "book_peak": 7.50, "book_offpeak": 4.00, "book_night": 4.00, "is_hala": True},
    "Eco Friendly": {"base_day": 5.00, "base_night": 5.50, "per_km": 2.60, "per_min": 0.42, "min_fare": 15.0, "book_peak": 5.00, "book_offpeak": 3.50, "book_night": 3.00, "is_hala": False},
    "Electric": {"base_day": 5.00, "base_night": 5.50, "per_km": 2.80, "per_min": 0.45, "min_fare": 16.0, "book_peak": 5.50, "book_offpeak": 4.00, "book_night": 3.50, "is_hala": False},
    "Kids": {"base_day": 5.00, "base_night": 5.50, "per_km": 2.80, "per_min": 0.45, "min_fare": 18.0, "book_peak": 6.00, "book_offpeak": 4.50, "book_night": 3.50, "is_hala": False},
    "Hala Kids": {"base_day": 5.00, "base_night": 5.50, "per_km": 2.30, "per_min": 0.50, "min_fare": 15.0, "book_peak": 7.50, "book_offpeak": 4.50, "book_night": 4.00, "is_hala": True},
    "Premier": {"base_day": 5.00, "base_night": 5.50, "per_km": 4.50, "per_min": 0.80, "min_fare": 30.0, "book_peak": 8.00, "book_offpeak": 6.00, "book_night": 5.00, "is_hala": False},
    "MAX": {"base_day": 5.00, "base_night": 5.50, "per_km": 3.80, "per_min": 0.65, "min_fare": 25.0, "book_peak": 7.00, "book_offpeak": 5.00, "book_night": 4.50, "is_hala": False},
    "Hala MAX": {"base_day": 5.00, "base_night": 5.50, "per_km": 2.80, "per_min": 0.60, "min_fare": 20.0, "book_peak": 8.00, "book_offpeak": 5.50, "book_night": 5.00, "is_hala": True},
}

PAYMENT_METHODS = ["Credit Card", "Cash", "Careem Pay", "Careem Plus"]
ZONE_NAMES = list(ZONES)
PRODUCT_NAMES = list(PRODUCT_SPECS)
RAMADAN_START = pd.Timestamp("2025-03-01")
RAMADAN_END = pd.Timestamp("2025-03-29")
UAE_HOLIDAYS = {
    "2025-01-01",
    "2025-03-30",
    "2025-03-31",
    "2025-04-01",
    "2025-04-02",
    "2025-04-03",
    "2025-06-05",
    "2025-06-06",
    "2025-06-07",
    "2025-06-08",
    "2025-06-09",
    "2025-06-26",
    "2025-09-04",
    "2025-12-02",
    "2025-12-03",
}


def stable_rng(*parts: object) -> np.random.Generator:
    digest = hashlib.sha256("||".join(map(str, parts)).encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16) % (2 ** 32)
    return np.random.default_rng(seed)


def normalize_calendar_date(ride_dt: datetime | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(ride_dt)
    return pd.Timestamp(year=2025, month=timestamp.month, day=timestamp.day, hour=timestamp.hour, minute=timestamp.minute)


def get_distance_km(pickup_zone: str, dropoff_zone: str) -> float:
    rng = stable_rng("distance", pickup_zone, dropoff_zone)
    if pickup_zone == dropoff_zone:
        return round(float(rng.uniform(1.8, 4.2)), 2)

    key = (pickup_zone, dropoff_zone)
    if key not in DIST_MATRIX:
        key = (dropoff_zone, pickup_zone)
    base_distance = DIST_MATRIX.get(key, 15.0)
    return round(float(base_distance * rng.uniform(0.95, 1.05)), 2)


def get_salik_gates(pickup_zone: str, dropoff_zone: str) -> int:
    key = (pickup_zone, dropoff_zone)
    if key not in SALIK:
        key = (dropoff_zone, pickup_zone)
    return int(SALIK.get(key, 0))


def get_event_context(ride_dt: datetime | pd.Timestamp) -> dict[str, object]:
    calendar_dt = normalize_calendar_date(ride_dt).normalize()
    for event in EVENTS:
        start = pd.Timestamp(event["start"])
        end = pd.Timestamp(event["end"])
        if start <= calendar_dt <= end:
            return {
                "active_event": event["name"],
                "event_type": event["type"],
                "event_demand_multiplier": float(event["dmult"]),
            }
    return {"active_event": "None", "event_type": "None", "event_demand_multiplier": 1.0}


def get_time_context(ride_dt: datetime | pd.Timestamp) -> dict[str, object]:
    actual_dt = pd.Timestamp(ride_dt)
    calendar_dt = normalize_calendar_date(ride_dt)
    hour = int(actual_dt.hour)
    dow = int(actual_dt.dayofweek)
    is_weekend = dow in (4, 5)
    is_peak_hour = hour >= 16 if dow == 4 else ((8 <= hour < 10) or (16 <= hour < 20))
    is_late_night = hour >= 22 or hour < 6
    is_offpeak = not is_peak_hour and not is_late_night
    is_ramadan = bool(RAMADAN_START <= calendar_dt.normalize() <= RAMADAN_END)
    is_suhoor = bool(is_ramadan and 1 <= hour <= 3)
    is_iftar = bool(is_ramadan and hour == 17)
    is_public_holiday = calendar_dt.strftime("%Y-%m-%d") in UAE_HOLIDAYS
    return {
        "timestamp": actual_dt,
        "date": actual_dt.date().isoformat(),
        "hour": hour,
        "minute": int(actual_dt.minute),
        "day_of_week": dow,
        "day_name": actual_dt.day_name(),
        "week_of_year": int(actual_dt.isocalendar().week),
        "month": int(actual_dt.month),
        "month_name": actual_dt.month_name(),
        "quarter": int(actual_dt.quarter),
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak_hour,
        "is_late_night": is_late_night,
        "is_offpeak": is_offpeak,
        "is_ramadan": is_ramadan,
        "is_uae_public_holiday": is_public_holiday,
        "is_suhoor_window": is_suhoor,
        "is_iftar_window": is_iftar,
    }


def build_trip_record(
    pickup_zone: str,
    dropoff_zone: str,
    product_type: str,
    ride_dt: datetime | pd.Timestamp,
    weather: dict[str, object],
    payment_method: str,
) -> dict[str, object]:
    time_context = get_time_context(ride_dt)
    event_context = get_event_context(ride_dt)
    pickup_meta = ZONES[pickup_zone]
    dropoff_meta = ZONES[dropoff_zone]
    rng = stable_rng("scenario", pickup_zone, dropoff_zone, product_type, time_context["timestamp"].isoformat(), payment_method)

    route_distance_km = get_distance_km(pickup_zone, dropoff_zone)
    salik_gates = get_salik_gates(pickup_zone, dropoff_zone)
    salik_cost_aed = round(salik_gates * 4.0, 2)
    is_airport_ride = pickup_zone == "DXB Airport" or dropoff_zone == "DXB Airport"
    is_intrazone_trip = pickup_zone == dropoff_zone
    is_hala_product = bool(PRODUCT_SPECS[product_type]["is_hala"])
    is_careem_plus = payment_method == "Careem Plus"

    temporal_demand = 1.18 if time_context["is_peak_hour"] else (0.92 if time_context["is_late_night"] else 1.00)
    ramadan_demand = 1.35 if time_context["is_iftar_window"] else (1.15 if time_context["is_suhoor_window"] else (0.96 if time_context["is_ramadan"] else 1.00))
    weekend_demand = 1.05 if time_context["is_weekend"] else 1.00
    demand_index = float(
        np.clip(
            pickup_meta["dmult"]
            * float(event_context["event_demand_multiplier"])
            * float(weather["weather_demand_factor"])
            * temporal_demand
            * ramadan_demand
            * weekend_demand,
            0.75,
            3.00,
        )
    )

    captain_availability_score = float(np.clip(1.0 - 0.32 * (demand_index - 1.0) + rng.normal(0, 0.06), 0.15, 1.0))
    supply_pressure_index = float(np.clip(1.0 - captain_availability_score, 0.0, 1.0))

    wait_base = 5.5 if is_airport_ride else (4.5 if time_context["is_peak_hour"] else 3.0)
    wait_time_min = float(np.clip(wait_base * (1.5 - captain_availability_score) + rng.uniform(0.4, 2.2), 1.0, 25.0))

    speed_low, speed_high = (22, 30) if time_context["is_peak_hour"] else ((48, 65) if time_context["is_late_night"] else (32, 48))
    avg_speed_kmh = float(rng.uniform(speed_low, speed_high))
    trip_duration_min = float((route_distance_km / max(avg_speed_kmh, 1.0)) * 60)

    product = PRODUCT_SPECS[product_type]
    flagfall = product["base_night"] if time_context["is_late_night"] else product["base_day"]
    booking_fee_aed = (
        product["book_peak"]
        if time_context["is_peak_hour"]
        else (product["book_night"] if time_context["is_late_night"] else product["book_offpeak"])
    )

    if is_hala_product:
        start_charge = 25.0 if pickup_zone == "DXB Airport" else flagfall + booking_fee_aed
        metered_fare_aed = start_charge + product["per_km"] * route_distance_km + wait_time_min * 0.50 + salik_cost_aed
        surge_multiplier = 1.0
        final_price_aed = max(metered_fare_aed, product["min_fare"])
    else:
        demand_supply_gap = float(np.clip(demand_index - captain_availability_score, 0.0, 1.5))
        surge_multiplier = float(np.clip(1.0 + demand_supply_gap * 0.55, 1.0, 2.5))
        metered_fare_aed = flagfall + booking_fee_aed + (product["per_km"] * route_distance_km + product["per_min"] * trip_duration_min) * surge_multiplier + salik_cost_aed
        final_price_aed = max(metered_fare_aed, product["min_fare"])

    pickup_lat = pickup_meta["lat"] + rng.normal(0, 0.003)
    pickup_lon = pickup_meta["lon"] + rng.normal(0, 0.003)
    dropoff_lat = dropoff_meta["lat"] + rng.normal(0, 0.003)
    dropoff_lon = dropoff_meta["lon"] + rng.normal(0, 0.003)

    return {
        "ride_id": "SIMULATED-RIDE",
        "customer_id": "SIM-CUSTOMER",
        "captain_id": "SIM-CAPTAIN",
        "timestamp": time_context["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
        "date": time_context["date"],
        "hour": time_context["hour"],
        "minute": time_context["minute"],
        "day_of_week": time_context["day_of_week"],
        "day_name": time_context["day_name"],
        "week_of_year": time_context["week_of_year"],
        "month": time_context["month"],
        "month_name": time_context["month_name"],
        "quarter": time_context["quarter"],
        "is_weekend": time_context["is_weekend"],
        "is_peak_hour": time_context["is_peak_hour"],
        "is_late_night": time_context["is_late_night"],
        "is_offpeak": time_context["is_offpeak"],
        "is_ramadan": time_context["is_ramadan"],
        "is_uae_public_holiday": time_context["is_uae_public_holiday"],
        "is_suhoor_window": time_context["is_suhoor_window"],
        "is_iftar_window": time_context["is_iftar_window"],
        "active_event": event_context["active_event"],
        "event_type": event_context["event_type"],
        "event_demand_multiplier": round(float(event_context["event_demand_multiplier"]), 3),
        "temperature_c": round(float(weather["temperature_c"]), 1),
        "humidity_pct": round(float(weather["humidity_pct"]), 1),
        "is_rain": bool(weather["is_rain"]),
        "is_sandstorm": bool(weather["is_sandstorm"]),
        "weather_demand_factor": round(float(weather["weather_demand_factor"]), 3),
        "pickup_zone": pickup_zone,
        "dropoff_zone": dropoff_zone,
        "pickup_lat": round(float(pickup_lat), 6),
        "pickup_lon": round(float(pickup_lon), 6),
        "dropoff_lat": round(float(dropoff_lat), 6),
        "dropoff_lon": round(float(dropoff_lon), 6),
        "pickup_area_type": pickup_meta["tier"],
        "dropoff_area_type": dropoff_meta["tier"],
        "is_airport_ride": is_airport_ride,
        "is_intrazone_trip": is_intrazone_trip,
        "route_distance_km": round(route_distance_km, 2),
        "salik_gates": salik_gates,
        "salik_cost_aed": round(salik_cost_aed, 2),
        "product_type": product_type,
        "is_hala_product": is_hala_product,
        "payment_method": payment_method,
        "is_careem_plus": is_careem_plus,
        "demand_index": round(demand_index, 3),
        "captain_availability_score": round(captain_availability_score, 3),
        "supply_pressure_index": round(supply_pressure_index, 3),
        "wait_time_min": round(wait_time_min, 1),
        "trip_duration_min": round(trip_duration_min, 1),
        "avg_speed_kmh": round(avg_speed_kmh, 1),
        "surge_multiplier": round(surge_multiplier, 3),
        "booking_fee_aed": round(float(booking_fee_aed), 2),
        "metered_fare_aed": round(float(metered_fare_aed), 2),
        "final_price_aed": round(float(final_price_aed), 2),
        "price_per_km_aed": round(float(final_price_aed / max(route_distance_km, 0.1)), 2),
        "booking_status": "Completed",
        "cancellation_reason": "N/A",
        "captain_rating": 4.8,
        "customer_rating": 4.7,
        "eta_deviation_min": 0.0,
        "weather_source": weather.get("source", "Unknown"),
        "weather_label": weather.get("weather_label", "Clear"),
    }


def build_inference_frame(record: dict[str, object], feature_columns: list[str]):
    frame = pd.DataFrame([record])
    X_row, _, _ = prepare_inference_frame(frame, feature_columns)
    return X_row, frame