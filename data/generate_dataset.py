#!/usr/bin/env python3
"""
Dubai Ride-Hailing Mirror Dataset Generator
===========================================
XPrice Project — MIT 622 Data Analytics for Managers
Group 1: Mohammadsadegh Solouki · Artin Fateh Basharzad · Fatema Alblooshi

Generates a synthetic mirror dataset (~160,000 records) of Careem ride-hailing
operations in Dubai, 2025. All parameters are calibrated to publicly documented
operational data (Careem Engineering Blog, e& FY2025 Report, WTW 2024 MENA
Ride-Hailing Report, Dubai Events Calendar, and UAE historical weather profiles).

This is a researcher-constructed dataset used to demonstrate the XPrice XAI
framework — it is not proprietary Careem operational data.

Usage:
    python generate_dataset.py

Output:
    data/processed/dubai_rides_2025.csv
    data/DATA_DICTIONARY.md  (auto-generated)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import json
import os
import time

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ═════════════════════════════════════════════════════════════════════════════
# 1. DUBAI ZONE DEFINITIONS
#    16 zones covering Dubai's main operational areas.
#    Coordinates are real centroids; demand_tier reflects commercial intensity.
# ═════════════════════════════════════════════════════════════════════════════
ZONES = {
    "Downtown Dubai": {
        "lat": 25.1972, "lon": 55.2796,
        "demand_tier": "very_high",
        "area_type": "tourist_commercial",
        "base_captain_density": 0.92,
        "notes": "Burj Khalifa, Dubai Mall, Fountain — highest tourist footfall"
    },
    "Business Bay": {
        "lat": 25.1865, "lon": 55.2628,
        "demand_tier": "high",
        "area_type": "commercial",
        "base_captain_density": 0.85,
        "notes": "Dense corporate towers, Mon–Fri corporate commute dominant"
    },
    "DIFC": {
        "lat": 25.2118, "lon": 55.2826,
        "demand_tier": "high",
        "area_type": "financial_commercial",
        "base_captain_density": 0.82,
        "notes": "Dubai International Financial Centre — high-value business rides"
    },
    "Dubai Marina": {
        "lat": 25.0763, "lon": 55.1303,
        "demand_tier": "very_high",
        "area_type": "tourist_residential",
        "base_captain_density": 0.90,
        "notes": "Upscale waterfront — strong weekend/evening demand"
    },
    "JBR": {
        "lat": 25.0774, "lon": 55.1296,
        "demand_tier": "high",
        "area_type": "tourist_beach",
        "base_captain_density": 0.85,
        "notes": "Jumeirah Beach Residence — beach, restaurants, tourist strip"
    },
    "Palm Jumeirah": {
        "lat": 25.1124, "lon": 55.1390,
        "demand_tier": "high",
        "area_type": "luxury_residential",
        "base_captain_density": 0.75,
        "notes": "Luxury island — higher fares, lower volume, Atlantis trips"
    },
    "Jumeirah": {
        "lat": 25.2084, "lon": 55.2450,
        "demand_tier": "medium",
        "area_type": "residential",
        "base_captain_density": 0.70,
        "notes": "Expat residential — steady but lower-density demand"
    },
    "Deira": {
        "lat": 25.2631, "lon": 55.3246,
        "demand_tier": "high",
        "area_type": "commercial_old_city",
        "base_captain_density": 0.85,
        "notes": "Gold Souk, Spice Souk, Port — high South Asian commuter demand"
    },
    "Bur Dubai": {
        "lat": 25.2532, "lon": 55.3000,
        "demand_tier": "medium",
        "area_type": "mixed_old_city",
        "base_captain_density": 0.80,
        "notes": "Historic district, mixed residential/commercial"
    },
    "Dubai Airport (DXB)": {
        "lat": 25.2532, "lon": 55.3657,
        "demand_tier": "very_high",
        "area_type": "airport",
        "base_captain_density": 0.95,
        "notes": "Dubai International Airport — consistent 24/7 demand"
    },
    "Al Quoz": {
        "lat": 25.1548, "lon": 55.2347,
        "demand_tier": "low",
        "area_type": "industrial",
        "base_captain_density": 0.50,
        "notes": "Industrial/warehouse district — low demand, captains reluctant"
    },
    "Dubai Hills": {
        "lat": 25.0912, "lon": 55.2400,
        "demand_tier": "medium",
        "area_type": "residential_suburban",
        "base_captain_density": 0.65,
        "notes": "Newer suburban residential — growing demand"
    },
    "Al Barsha": {
        "lat": 25.1059, "lon": 55.2015,
        "demand_tier": "medium",
        "area_type": "residential_commercial",
        "base_captain_density": 0.72,
        "notes": "Mall of the Emirates area — steady mid-range demand"
    },
    "Mirdif": {
        "lat": 25.2197, "lon": 55.4144,
        "demand_tier": "medium",
        "area_type": "residential_suburban",
        "base_captain_density": 0.60,
        "notes": "Eastern residential suburb — longer trips to city core"
    },
    "Karama": {
        "lat": 25.2375, "lon": 55.3057,
        "demand_tier": "medium",
        "area_type": "residential_commercial",
        "base_captain_density": 0.75,
        "notes": "Dense South Asian expat area — high cash payment rate"
    },
    "Sharjah Border": {
        "lat": 25.3463, "lon": 55.4213,
        "demand_tier": "low",
        "area_type": "border_transit",
        "base_captain_density": 0.55,
        "notes": "Inter-emirate transit zone — long trips, lower frequency"
    },
}

ZONE_NAMES = list(ZONES.keys())

# Zone demand multipliers (relative to city average = 1.0)
ZONE_DEMAND_MULT = {
    "Downtown Dubai":    1.38,
    "Business Bay":      1.15,
    "DIFC":              1.10,
    "Dubai Marina":      1.30,
    "JBR":               1.22,
    "Palm Jumeirah":     1.05,
    "Jumeirah":          0.85,
    "Deira":             1.20,
    "Bur Dubai":         1.00,
    "Dubai Airport (DXB)": 1.42,
    "Al Quoz":           0.52,
    "Dubai Hills":       0.75,
    "Al Barsha":         0.90,
    "Mirdif":            0.70,
    "Karama":            0.96,
    "Sharjah Border":    0.58,
}

# ═════════════════════════════════════════════════════════════════════════════
# 2. CAREEM PRODUCTS — Dubai 2025 fare structure
#    Calibrated to Careem's published rates and RTA metered taxi rates.
# ═════════════════════════════════════════════════════════════════════════════
PRODUCTS = {
    "Careem Go": {
        "base_fare": 5.00, "per_km": 1.49, "per_min": 0.25, "min_fare": 12.00,
        "share": 0.44, "accepts_cash": True, "segment": "budget",
        "notes": "Entry-level ride — highest volume product"
    },
    "Careem Go+": {
        "base_fare": 6.00, "per_km": 1.75, "per_min": 0.30, "min_fare": 15.00,
        "share": 0.22, "accepts_cash": True, "segment": "standard",
        "notes": "Mid-tier comfort upgrade"
    },
    "Careem Business": {
        "base_fare": 12.00, "per_km": 2.80, "per_min": 0.55, "min_fare": 28.00,
        "share": 0.10, "accepts_cash": False, "segment": "premium",
        "notes": "Corporate/premium — highest AED per ride, no cash"
    },
    "Hala Taxi": {
        "base_fare": 12.00, "per_km": 1.97, "per_min": 0.37, "min_fare": 12.00,
        "share": 0.16, "accepts_cash": True, "segment": "standard",
        "notes": "RTA-licensed metered taxi via Careem app"
    },
    "Hala EV": {
        "base_fare": 7.00, "per_km": 1.65, "per_min": 0.28, "min_fare": 14.00,
        "share": 0.08, "accepts_cash": False, "segment": "eco",
        "notes": "Electric vehicle option — growing fleet in 2025"
    },
}
PRODUCT_NAMES = list(PRODUCTS.keys())
PRODUCT_SHARES = [PRODUCTS[p]["share"] for p in PRODUCT_NAMES]

# ═════════════════════════════════════════════════════════════════════════════
# 3. DUBAI EVENTS CALENDAR 2025
#    Source: Dubai Tourism, GITEX, Dubai Events official calendars
# ═════════════════════════════════════════════════════════════════════════════
EVENTS_2025 = [
    {
        "name": "Dubai Shopping Festival (DSF)",
        "start": date(2025, 1, 16), "end": date(2025, 2, 16),
        "venue_zones": ["Downtown Dubai", "Deira", "Dubai Marina", "Al Barsha"],
        "demand_multiplier": 1.35, "type": "shopping_festival",
        "notes": "City-wide retail festival, heavy footfall in malls and souks"
    },
    {
        "name": "Dubai Food Festival",
        "start": date(2025, 2, 21), "end": date(2025, 3, 8),
        "venue_zones": ["JBR", "Dubai Marina", "DIFC", "Business Bay"],
        "demand_multiplier": 1.20, "type": "food_festival",
        "notes": "Restaurant and beach pop-up events"
    },
    {
        "name": "Ramadan",
        "start": date(2025, 3, 1), "end": date(2025, 3, 29),
        "venue_zones": [],  # city-wide cultural shift
        "demand_multiplier": 0.85, "type": "religious",
        "notes": "Daytime demand drops; Iftar surge 17:30-19:00; Suhoor 01:00-03:00"
    },
    {
        "name": "Eid Al Fitr",
        "start": date(2025, 3, 30), "end": date(2025, 4, 2),
        "venue_zones": ["Downtown Dubai", "Dubai Marina", "JBR", "Deira", "Bur Dubai"],
        "demand_multiplier": 1.55, "type": "public_holiday",
        "notes": "Major holiday — family outings, mall visits, airport travel"
    },
    {
        "name": "Eid Al Adha",
        "start": date(2025, 6, 6), "end": date(2025, 6, 9),
        "venue_zones": ["Deira", "Bur Dubai", "Karama", "Downtown Dubai"],
        "demand_multiplier": 1.42, "type": "public_holiday",
        "notes": "Second major holiday — family gatherings and travel"
    },
    {
        "name": "Dubai Summer Surprises",
        "start": date(2025, 7, 1), "end": date(2025, 9, 6),
        "venue_zones": ["Downtown Dubai", "Al Barsha", "Deira"],
        "demand_multiplier": 0.82, "type": "summer_festival",
        "notes": "Indoor mall-based events; heat reduces overall outdoor mobility"
    },
    {
        "name": "GITEX Global",
        "start": date(2025, 10, 13), "end": date(2025, 10, 17),
        "venue_zones": ["DIFC", "Business Bay", "Bur Dubai", "Dubai Airport (DXB)"],
        "demand_multiplier": 1.62, "type": "tech_conference",
        "notes": "World's largest tech show — 100,000+ visitors, massive corporate demand"
    },
    {
        "name": "Dubai Airshow",
        "start": date(2025, 11, 17), "end": date(2025, 11, 21),
        "venue_zones": ["Dubai Airport (DXB)", "Business Bay", "DIFC"],
        "demand_multiplier": 1.45, "type": "exhibition",
        "notes": "International aerospace show — business travel surge"
    },
    {
        "name": "UAE National Day",
        "start": date(2025, 12, 1), "end": date(2025, 12, 3),
        "venue_zones": ["Downtown Dubai", "Dubai Marina", "JBR", "Jumeirah"],
        "demand_multiplier": 1.72, "type": "national_holiday",
        "notes": "Largest national celebration — fireworks, corniche events, high traffic"
    },
    {
        "name": "New Year's Eve",
        "start": date(2025, 12, 31), "end": date(2025, 12, 31),
        "venue_zones": ["Downtown Dubai", "Palm Jumeirah", "Dubai Marina", "JBR"],
        "demand_multiplier": 2.25, "type": "celebration",
        "notes": "Burj Khalifa fireworks — single highest-surge night of the year"
    },
]

# UAE Public Holidays 2025 (official)
UAE_HOLIDAYS = {
    date(2025, 1, 1),   # New Year's Day
    date(2025, 3, 30),  # Eid Al Fitr — Day 1
    date(2025, 3, 31),  # Eid Al Fitr — Day 2
    date(2025, 4, 1),   # Eid Al Fitr — Day 3
    date(2025, 6, 6),   # Eid Al Adha — Day 1
    date(2025, 6, 7),   # Eid Al Adha — Day 2
    date(2025, 6, 8),   # Eid Al Adha — Day 3
    date(2025, 6, 9),   # Eid Al Adha — Day 4
    date(2025, 6, 27),  # Islamic New Year
    date(2025, 9, 5),   # Prophet's Birthday
    date(2025, 12, 1),  # Commemoration Day
    date(2025, 12, 2),  # UAE National Day
    date(2025, 12, 3),  # UAE National Day holiday
}

RAMADAN_DATES = (date(2025, 3, 1), date(2025, 3, 29))

# ═════════════════════════════════════════════════════════════════════════════
# 4. DUBAI WEATHER PROFILES (monthly)
#    Source: UAE Meteorological Authority historical averages
# ═════════════════════════════════════════════════════════════════════════════
WEATHER_BY_MONTH = {
    1:  {"temp_mu": 22, "temp_sd": 3.0, "hum_mu": 65, "rain_p": 0.050, "storm_p": 0.018},
    2:  {"temp_mu": 24, "temp_sd": 3.0, "hum_mu": 63, "rain_p": 0.040, "storm_p": 0.020},
    3:  {"temp_mu": 28, "temp_sd": 3.0, "hum_mu": 60, "rain_p": 0.035, "storm_p": 0.025},
    4:  {"temp_mu": 34, "temp_sd": 3.0, "hum_mu": 55, "rain_p": 0.010, "storm_p": 0.035},
    5:  {"temp_mu": 39, "temp_sd": 2.5, "hum_mu": 50, "rain_p": 0.002, "storm_p": 0.050},
    6:  {"temp_mu": 42, "temp_sd": 2.0, "hum_mu": 60, "rain_p": 0.000, "storm_p": 0.060},
    7:  {"temp_mu": 43, "temp_sd": 1.8, "hum_mu": 65, "rain_p": 0.000, "storm_p": 0.045},
    8:  {"temp_mu": 44, "temp_sd": 1.8, "hum_mu": 70, "rain_p": 0.000, "storm_p": 0.038},
    9:  {"temp_mu": 41, "temp_sd": 2.0, "hum_mu": 68, "rain_p": 0.008, "storm_p": 0.030},
    10: {"temp_mu": 36, "temp_sd": 2.5, "hum_mu": 60, "rain_p": 0.028, "storm_p": 0.025},
    11: {"temp_mu": 30, "temp_sd": 3.0, "hum_mu": 60, "rain_p": 0.040, "storm_p": 0.020},
    12: {"temp_mu": 24, "temp_sd": 3.0, "hum_mu": 62, "rain_p": 0.048, "storm_p": 0.018},
}

# ═════════════════════════════════════════════════════════════════════════════
# 5. HOURLY DEMAND PROFILES
#    Weights represent relative ride volume at each hour (normalized to peak=1.0)
# ═════════════════════════════════════════════════════════════════════════════
# UAE: weekend = Friday + Saturday
DEMAND_WEEKDAY = {
    0: 0.28,  1: 0.38,  2: 0.42,  3: 0.22,  4: 0.14,  5: 0.22,
    6: 0.48,  7: 0.92,  8: 1.00,  9: 0.86, 10: 0.60, 11: 0.55,
   12: 0.72, 13: 0.78, 14: 0.52, 15: 0.45, 16: 0.72, 17: 0.96,
   18: 1.00, 19: 0.92, 20: 0.82, 21: 0.76, 22: 0.65, 23: 0.50,
}
DEMAND_WEEKEND = {
    0: 0.62,  1: 0.72,  2: 0.65,  3: 0.42,  4: 0.20,  5: 0.15,
    6: 0.25,  7: 0.38,  8: 0.50,  9: 0.60, 10: 0.75, 11: 0.82,
   12: 0.88, 13: 0.82, 14: 0.72, 15: 0.68, 16: 0.78, 17: 0.88,
   18: 0.96, 19: 1.00, 20: 0.96, 21: 0.92, 22: 0.88, 23: 0.78,
}
# Ramadan: daytime suppressed; pre-Iftar (17:00) spikes; Suhoor (01-03) rises
DEMAND_RAMADAN = {
    0: 0.52,  1: 0.65,  2: 0.55,  3: 0.28,  4: 0.48,  5: 0.32,
    6: 0.22,  7: 0.42,  8: 0.55,  9: 0.50, 10: 0.40, 11: 0.35,
   12: 0.28, 13: 0.25, 14: 0.20, 15: 0.22, 16: 0.38, 17: 1.00,
   18: 0.92, 19: 0.72, 20: 0.68, 21: 0.72, 22: 0.68, 23: 0.62,
}

# ═════════════════════════════════════════════════════════════════════════════
# 6. ROAD DISTANCE MATRIX (km, calibrated to Google Maps road routing)
#    Source: manually verified against Google Maps, January 2025
# ═════════════════════════════════════════════════════════════════════════════
_KNOWN = {
    ("Downtown Dubai",      "Business Bay"):             3.5,
    ("Downtown Dubai",      "DIFC"):                     2.8,
    ("Downtown Dubai",      "Jumeirah"):                  8.5,
    ("Downtown Dubai",      "Bur Dubai"):                 5.5,
    ("Downtown Dubai",      "Deira"):                    10.5,
    ("Downtown Dubai",      "Dubai Airport (DXB)"):      17.0,
    ("Downtown Dubai",      "Dubai Marina"):             32.0,
    ("Downtown Dubai",      "JBR"):                      33.5,
    ("Downtown Dubai",      "Palm Jumeirah"):             29.0,
    ("Downtown Dubai",      "Al Quoz"):                  10.0,
    ("Downtown Dubai",      "Dubai Hills"):              19.5,
    ("Downtown Dubai",      "Al Barsha"):                23.0,
    ("Downtown Dubai",      "Mirdif"):                   22.5,
    ("Downtown Dubai",      "Karama"):                    7.0,
    ("Downtown Dubai",      "Sharjah Border"):           31.0,
    ("Business Bay",        "DIFC"):                      2.5,
    ("Business Bay",        "Jumeirah"):                  9.0,
    ("Business Bay",        "Bur Dubai"):                 6.0,
    ("Business Bay",        "Deira"):                    13.0,
    ("Business Bay",        "Dubai Airport (DXB)"):      18.5,
    ("Business Bay",        "Dubai Marina"):             29.0,
    ("Business Bay",        "JBR"):                      30.5,
    ("Business Bay",        "Palm Jumeirah"):             28.0,
    ("Business Bay",        "Al Quoz"):                   8.0,
    ("Business Bay",        "Dubai Hills"):              17.0,
    ("Business Bay",        "Al Barsha"):                21.0,
    ("Business Bay",        "Karama"):                    5.5,
    ("Business Bay",        "Mirdif"):                   22.0,
    ("DIFC",                "Jumeirah"):                  9.0,
    ("DIFC",                "Bur Dubai"):                 7.0,
    ("DIFC",                "Deira"):                    14.5,
    ("DIFC",                "Dubai Airport (DXB)"):      16.5,
    ("DIFC",                "Dubai Marina"):             31.0,
    ("DIFC",                "JBR"):                      32.0,
    ("DIFC",                "Al Quoz"):                   9.0,
    ("Dubai Marina",        "JBR"):                       2.5,
    ("Dubai Marina",        "Palm Jumeirah"):              8.5,
    ("Dubai Marina",        "Al Barsha"):                10.5,
    ("Dubai Marina",        "Dubai Hills"):              13.5,
    ("Dubai Marina",        "Al Quoz"):                  20.0,
    ("Dubai Marina",        "Jumeirah"):                 18.5,
    ("Dubai Marina",        "Bur Dubai"):                36.0,
    ("Dubai Marina",        "Deira"):                    48.0,
    ("Dubai Marina",        "Dubai Airport (DXB)"):      46.5,
    ("Dubai Marina",        "Karama"):                   33.0,
    ("Dubai Marina",        "Mirdif"):                   52.0,
    ("Dubai Marina",        "Sharjah Border"):           58.0,
    ("JBR",                 "Palm Jumeirah"):              7.0,
    ("JBR",                 "Al Barsha"):                12.0,
    ("JBR",                 "Dubai Hills"):              15.0,
    ("Palm Jumeirah",       "Jumeirah"):                 12.0,
    ("Palm Jumeirah",       "Al Barsha"):                14.5,
    ("Palm Jumeirah",       "Downtown Dubai"):           29.0,
    ("Deira",               "Bur Dubai"):                 7.5,
    ("Deira",               "Dubai Airport (DXB)"):       6.5,
    ("Deira",               "Mirdif"):                   16.5,
    ("Deira",               "Karama"):                    9.0,
    ("Deira",               "Sharjah Border"):           13.0,
    ("Bur Dubai",           "Karama"):                    3.5,
    ("Bur Dubai",           "Dubai Airport (DXB)"):      12.5,
    ("Bur Dubai",           "Mirdif"):                   19.5,
    ("Dubai Airport (DXB)", "Mirdif"):                    9.0,
    ("Dubai Airport (DXB)", "Karama"):                   13.0,
    ("Dubai Airport (DXB)", "Sharjah Border"):           16.5,
    ("Dubai Airport (DXB)", "Deira"):                     6.5,
    ("Al Quoz",             "Al Barsha"):                 7.5,
    ("Al Quoz",             "Dubai Hills"):               9.0,
    ("Al Quoz",             "Jumeirah"):                  9.5,
    ("Al Barsha",           "Dubai Hills"):               6.0,
    ("Al Barsha",           "Jumeirah"):                 11.0,
    ("Mirdif",              "Sharjah Border"):           10.5,
    ("Karama",              "Bur Dubai"):                 3.5,
}


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((phi2 - phi1) / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin((np.radians(lon2 - lon1)) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def road_distance(zone_a: str, zone_b: str) -> float:
    """Return road distance in km between two zones."""
    if zone_a == zone_b:
        return round(np.random.uniform(1.2, 4.8), 2)
    for key in [(zone_a, zone_b), (zone_b, zone_a)]:
        if key in _KNOWN:
            base = _KNOWN[key]
            # Add realistic route variation ±8%
            return round(base * np.random.uniform(0.93, 1.08), 2)
    # Fallback: haversine × road detour factor (Dubai avg ~1.42)
    sl = _haversine(ZONES[zone_a]["lat"], ZONES[zone_a]["lon"],
                    ZONES[zone_b]["lat"], ZONES[zone_b]["lon"])
    return round(sl * np.random.uniform(1.35, 1.52), 2)


# Pre-compute distance lookup matrix for dropoff zone sampling
def _build_dist_matrix():
    n = len(ZONE_NAMES)
    mat = np.zeros((n, n))
    for i, za in enumerate(ZONE_NAMES):
        for j, zb in enumerate(ZONE_NAMES):
            if i == j:
                mat[i, j] = 3.0
            else:
                k1, k2 = (za, zb), (zb, za)
                if k1 in _KNOWN:
                    mat[i, j] = _KNOWN[k1]
                elif k2 in _KNOWN:
                    mat[i, j] = _KNOWN[k2]
                else:
                    mat[i, j] = _haversine(
                        ZONES[za]["lat"], ZONES[za]["lon"],
                        ZONES[zb]["lat"], ZONES[zb]["lon"]
                    ) * 1.42
    return mat

DIST_MATRIX = _build_dist_matrix()
ZONE_IDX = {z: i for i, z in enumerate(ZONE_NAMES)}

# ═════════════════════════════════════════════════════════════════════════════
# 7. HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def get_weather(month: int) -> dict:
    w = WEATHER_BY_MONTH[month]
    temp = float(np.clip(np.random.normal(w["temp_mu"], w["temp_sd"]), 14, 48))
    hum  = float(np.clip(np.random.normal(w["hum_mu"], 8), 25, 95))
    rain  = bool(np.random.random() < w["rain_p"])
    storm = bool(np.random.random() < w["storm_p"]) if not rain else False
    factor = 1.0
    if rain:
        factor = float(np.random.uniform(1.32, 1.85))   # Rain = massive Dubai demand spike
    elif storm:
        factor = float(np.random.uniform(1.12, 1.42))
    elif temp > 41:
        factor = float(np.random.uniform(1.05, 1.22))   # Extreme heat → more rides
    return {
        "temperature_c":      round(temp, 1),
        "humidity_pct":       round(hum, 1),
        "is_rain":            rain,
        "is_sandstorm":       storm,
        "weather_demand_factor": round(factor, 3),
    }


def get_event_context(d: date, pickup_zone: str):
    """Return (event_name, event_type, zone_demand_mult) for a given date/zone."""
    active = [e for e in EVENTS_2025 if e["start"] <= d <= e["end"]]
    if not active:
        return "None", "None", 1.0
    best = max(active, key=lambda e: e["demand_multiplier"])
    if pickup_zone in best["venue_zones"]:
        mult = best["demand_multiplier"]
    else:
        # Partial city-wide lift
        mult = 1.0 + (best["demand_multiplier"] - 1.0) * 0.28
    return best["name"], best["type"], round(mult, 3)


def calc_surge(hour, is_weekend, is_ramadan, is_holiday,
               event_mult, weather_factor, zone_mult, captain_avail) -> float:
    """
    Compute surge multiplier [1.0, 2.5].
    Demand-supply gap drives surge; multiple real-world factors compound.
    """
    if is_ramadan:
        hw = DEMAND_RAMADAN[hour]
    elif is_weekend:
        hw = DEMAND_WEEKEND[hour]
    else:
        hw = DEMAND_WEEKDAY[hour]

    demand = hw * zone_mult * event_mult * weather_factor
    if is_holiday:
        demand *= 1.30

    # Supply tightness
    supply_tightness = 1.0 - captain_avail * 0.45

    # Imbalance → surge
    imbalance = max(0.0, demand / 0.72 - 1.0) + supply_tightness * 0.28
    surge = 1.0 + imbalance * 0.46

    # Ramadan pre-Iftar spike (17:00–18:00 is sharpest)
    if is_ramadan and hour in [17, 18]:
        surge *= np.random.uniform(1.18, 1.35)

    # New Year's Eve extra spike (23:00–01:00)
    if hour in [23, 0] and not is_ramadan:
        pass  # handled via event_mult already

    surge += np.random.normal(0, 0.04)
    return float(np.clip(round(surge, 3), 1.0, 2.50))


def calc_price(product: str, dist_km: float, dur_min: float, surge: float):
    """Returns (final_price_aed, metered_fare_aed)."""
    p = PRODUCTS[product]
    fare = p["base_fare"] + p["per_km"] * dist_km + p["per_min"] * dur_min
    fare = max(fare, p["min_fare"])
    final = round(fare * surge, 2)
    return final, round(fare, 2)


def sim_outcome(surge, wait_min, payment, dist_km, is_rain, hour, zone) -> str:
    """Simulate booking outcome based on operational factors."""
    p = 0.89  # Dubai baseline completion (highest-performing Careem market)
    if surge > 1.90:     p -= 0.10
    elif surge > 1.60:   p -= 0.05
    elif surge > 1.40:   p -= 0.02
    if payment == "Cash" and dist_km < 4.5:  p -= 0.14
    if payment == "Cash" and dist_km < 3.0:  p -= 0.08  # stacks
    if dist_km < 2.5:    p -= 0.06
    if wait_min > 9:     p -= 0.08
    if wait_min > 12:    p -= 0.06  # stacks
    if is_rain:          p -= 0.11
    if hour in [2, 3, 4]: p -= 0.06
    if zone == "Al Quoz": p -= 0.10
    if zone == "Sharjah Border": p -= 0.06
    p = float(np.clip(p, 0.52, 0.96))
    if np.random.random() < p:
        return "Completed"
    # Distribute failure modes contextually
    r = np.random.random()
    if payment == "Cash" and dist_km < 4.5:
        cuts = [0.55, 0.80]
        cats = ["Captain Cancelled", "No Captain Found", "Customer Cancelled"]
    elif wait_min > 9:
        cuts = [0.45, 0.72]
        cats = ["Customer Cancelled", "Captain Cancelled", "No Captain Found"]
    elif is_rain:
        cuts = [0.50, 0.75]
        cats = ["No Captain Found", "Captain Cancelled", "Customer Cancelled"]
    else:
        cuts = [0.40, 0.70]
        cats = ["Captain Cancelled", "No Captain Found", "Customer Cancelled"]
    if r < cuts[0]:  return cats[0]
    if r < cuts[1]:  return cats[1]
    return cats[2]


def cancel_reason(outcome, dist_km, payment, wait_min, surge) -> str:
    if outcome == "Completed":
        return "N/A"
    if outcome == "No Captain Found":
        return "No Captain Available"
    if outcome == "Customer Cancelled":
        if wait_min > 9:   return "Wait Time Too Long"
        if surge > 1.75:   return "Price Too High"
        return str(np.random.choice(
            ["Changed Plans", "Wait Time Too Long", "Found Alternative", "Price Too High"],
            p=[0.30, 0.35, 0.25, 0.10]))
    if outcome == "Captain Cancelled":
        if payment == "Cash" and dist_km < 4.5:
            return "Cash Trip – Short Distance"
        return str(np.random.choice(
            ["Ride Too Short / Uneconomical", "Wrong Pickup Location",
             "Cash Payment Refused", "Customer Unresponsive", "Personal Reason"],
            p=[0.34, 0.25, 0.20, 0.13, 0.08]))
    return "Unknown"


# ═════════════════════════════════════════════════════════════════════════════
# 8. MAIN DATASET GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate(target: int = 160_000, out: str = "data/processed/dubai_rides_2025.csv"):
    t0 = time.time()
    print("=" * 65)
    print("  XPrice — Dubai Ride Dataset Generator")
    print(f"  Target: {target:,} records | Year: 2025 | City: Dubai, UAE")
    print("=" * 65)

    all_dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(365)]
    base_per_day = target // 365
    rows = []
    ride_id = 1

    # Pre-compute zone demand weight array
    zdw = np.array([ZONE_DEMAND_MULT[z] for z in ZONE_NAMES], dtype=float)
    zdw /= zdw.sum()

    for d in all_dates:
        is_weekend  = d.weekday() in (4, 5)   # Fri = 4, Sat = 5 in UAE
        is_holiday  = d in UAE_HOLIDAYS
        is_ramadan  = RAMADAN_DATES[0] <= d <= RAMADAN_DATES[1]
        month       = d.month

        # Daily weather (constant for all rides that day)
        wx = get_weather(month)

        # City-level event context for volume calculation
        ev_name_city, ev_type_city, ev_mult_city = get_event_context(d, "Downtown Dubai")

        # Adjust daily ride volume
        vol = base_per_day
        if is_holiday:    vol = int(vol * 1.42)
        if is_weekend:    vol = int(vol * 1.14)
        if ev_name_city != "None" and ev_type_city != "religious":
            vol = int(vol * min(ev_mult_city, 1.55))
        if ev_name_city == "Ramadan":    vol = int(vol * 0.88)
        if wx["is_rain"]:                vol = int(vol * 1.28)
        if month in (7, 8):              vol = int(vol * 0.82)   # Summer drop
        vol = max(180, min(vol, 1600))

        # Hourly demand pool for the day
        if is_ramadan:
            hd = DEMAND_RAMADAN
        elif is_weekend:
            hd = DEMAND_WEEKEND
        else:
            hd = DEMAND_WEEKDAY
        h_pool = []
        for hh, ww in hd.items():
            h_pool.extend([hh] * max(1, int(ww * 100)))

        sampled_hours    = np.random.choice(h_pool, size=vol)
        pickup_zones_arr = np.random.choice(ZONE_NAMES, size=vol, p=zdw)
        products_arr     = np.random.choice(PRODUCT_NAMES, size=vol, p=PRODUCT_SHARES)

        for i in range(vol):
            hour        = int(sampled_hours[i])
            minute      = int(np.random.randint(0, 60))
            second      = int(np.random.randint(0, 60))
            ts          = datetime(d.year, d.month, d.day, hour, minute, second)
            pickup_zone = pickup_zones_arr[i]
            product     = products_arr[i]

            # ── Dropoff zone ─────────────────────────────────────────────
            pu_idx = ZONE_IDX[pickup_zone]
            dists  = DIST_MATRIX[pu_idx].copy()
            # Distance-decay × destination demand
            dz_demand = np.array([ZONE_DEMAND_MULT[z] for z in ZONE_NAMES])
            drop_probs = dz_demand / (dists + 1.5)
            drop_probs[pu_idx] *= 0.25   # Suppress intra-zone (still possible)
            drop_probs /= drop_probs.sum()
            dropoff_zone = str(np.random.choice(ZONE_NAMES, p=drop_probs))

            # ── Distance & duration ───────────────────────────────────────
            dist_km  = road_distance(pickup_zone, dropoff_zone)
            is_peak  = hour in (7, 8, 9, 17, 18, 19, 20)
            is_night = hour in (0, 1, 2, 3, 4, 5)
            if is_peak:
                speed = float(np.random.uniform(20, 36))
            elif is_night:
                speed = float(np.random.uniform(55, 82))
            else:
                speed = float(np.random.uniform(36, 60))
            dur_min = max(4.0, round((dist_km / speed) * 60, 1))

            # ── Zone-specific event context ───────────────────────────────
            ev_name, ev_type, ev_mult = get_event_context(d, pickup_zone)

            # ── Captain availability ──────────────────────────────────────
            cap_avail = ZONES[pickup_zone]["base_captain_density"]
            if is_peak:              cap_avail *= float(np.random.uniform(0.52, 0.80))
            if wx["is_rain"]:        cap_avail *= float(np.random.uniform(0.48, 0.70))
            if wx["is_sandstorm"]:   cap_avail *= float(np.random.uniform(0.65, 0.82))
            if is_ramadan and hour in (17, 18):
                cap_avail *= float(np.random.uniform(0.42, 0.65))
            cap_avail = float(round(np.clip(cap_avail, 0.18, 1.0), 3))

            # ── Wait time (VTAT) ─────────────────────────────────────────
            if cap_avail > 0.82:    wt_range = (1.5, 4.5)
            elif cap_avail > 0.65:  wt_range = (3.0, 6.5)
            elif cap_avail > 0.45:  wt_range = (5.0, 10.0)
            else:                   wt_range = (8.0, 17.0)
            wait_min = round(float(np.random.uniform(*wt_range)), 1)

            # ── Surge ────────────────────────────────────────────────────
            surge = calc_surge(
                hour, is_weekend, is_ramadan, is_holiday,
                ev_mult, wx["weather_demand_factor"],
                ZONE_DEMAND_MULT[pickup_zone], cap_avail
            )

            # ── Payment method ────────────────────────────────────────────
            accepts_cash = PRODUCTS[product]["accepts_cash"]
            if accepts_cash:
                pay_opts = ["Card", "Cash", "Careem Pay", "Careem Plus"]
                # Karama / Bur Dubai / Deira have higher cash rates
                if pickup_zone in ("Karama", "Bur Dubai", "Deira"):
                    pay_probs = [0.42, 0.32, 0.18, 0.08]
                else:
                    pay_probs = [0.52, 0.20, 0.20, 0.08]
            else:
                pay_opts  = ["Card", "Careem Pay", "Careem Plus"]
                pay_probs = [0.58, 0.30, 0.12]
            payment = str(np.random.choice(pay_opts, p=pay_probs))

            # ── Price ────────────────────────────────────────────────────
            final_price, metered_fare = calc_price(product, dist_km, dur_min, surge)

            # ── Outcome ──────────────────────────────────────────────────
            outcome = sim_outcome(surge, wait_min, payment, dist_km,
                                  wx["is_rain"], hour, pickup_zone)
            reason  = cancel_reason(outcome, dist_km, payment, wait_min, surge)

            # ── Ratings ──────────────────────────────────────────────────
            if outcome == "Completed":
                cap_rating = round(float(np.clip(np.random.normal(4.32, 0.48), 1, 5)), 1)
                cust_base  = 4.45
                if wait_min > 9:    cust_base -= 0.40
                if wait_min > 12:   cust_base -= 0.25
                if surge > 1.80:    cust_base -= 0.30
                if surge > 2.10:    cust_base -= 0.20
                cust_rating = round(float(np.clip(np.random.normal(cust_base, 0.52), 1, 5)), 1)
                eta_dev = round(float(np.random.normal(2.5 if (is_peak or wx["is_rain"]) else 0.5, 1.4)), 1)
            else:
                cap_rating = cust_rating = eta_dev = None

            # ── Coordinates (jitter around zone centroid) ────────────────
            pu_lat = round(ZONES[pickup_zone]["lat"]  + float(np.random.uniform(-0.020, 0.020)), 6)
            pu_lon = round(ZONES[pickup_zone]["lon"]  + float(np.random.uniform(-0.020, 0.020)), 6)
            do_lat = round(ZONES[dropoff_zone]["lat"] + float(np.random.uniform(-0.020, 0.020)), 6)
            do_lon = round(ZONES[dropoff_zone]["lon"] + float(np.random.uniform(-0.020, 0.020)), 6)

            rows.append({
                # ── Identifiers
                "ride_id":                  f"DXB-2025-{ride_id:07d}",
                "customer_id":              f"CUS{np.random.randint(1, 22001):05d}",
                "captain_id":               f"CAP{np.random.randint(1, 3201):04d}",

                # ── Temporal
                "timestamp":                ts.strftime("%Y-%m-%d %H:%M:%S"),
                "date":                     d.strftime("%Y-%m-%d"),
                "hour":                     hour,
                "minute":                   minute,
                "day_of_week":              d.weekday(),
                "day_name":                 d.strftime("%A"),
                "week_of_year":             d.isocalendar()[1],
                "month":                    month,
                "month_name":               d.strftime("%B"),
                "quarter":                  (month - 1) // 3 + 1,
                "is_weekend":               is_weekend,
                "is_peak_hour":             is_peak,
                "is_late_night":            hour in (0, 1, 2, 3, 4),
                "is_ramadan":               is_ramadan,
                "is_uae_public_holiday":    is_holiday,
                "is_suhoor_window":         is_ramadan and hour in (0, 1, 2, 3),
                "is_iftar_window":          is_ramadan and hour in (17, 18),

                # ── Event context
                "active_event":             ev_name,
                "event_type":               ev_type,
                "event_demand_multiplier":  ev_mult,

                # ── Weather
                "temperature_c":            wx["temperature_c"],
                "humidity_pct":             wx["humidity_pct"],
                "is_rain":                  wx["is_rain"],
                "is_sandstorm":             wx["is_sandstorm"],
                "weather_demand_factor":    wx["weather_demand_factor"],

                # ── Geography
                "pickup_zone":              pickup_zone,
                "dropoff_zone":             dropoff_zone,
                "pickup_lat":               pu_lat,
                "pickup_lon":               pu_lon,
                "dropoff_lat":              do_lat,
                "dropoff_lon":              do_lon,
                "pickup_area_type":         ZONES[pickup_zone]["area_type"],
                "dropoff_area_type":        ZONES[dropoff_zone]["area_type"],
                "is_airport_ride":          (
                    pickup_zone == "Dubai Airport (DXB)" or
                    dropoff_zone == "Dubai Airport (DXB)"
                ),
                "is_intrazone_trip":        pickup_zone == dropoff_zone,

                # ── Trip details
                "product_type":             product,
                "route_distance_km":        dist_km,
                "trip_duration_min":        dur_min,
                "avg_speed_kmh":            round(dist_km / (dur_min / 60), 1),
                "payment_method":           payment,
                "is_careem_plus":           payment == "Careem Plus",

                # ── Supply
                "captain_availability_score": cap_avail,
                "wait_time_min":            wait_min,

                # ── Pricing (TARGET for ML model)
                "surge_multiplier":         surge,
                "metered_fare_aed":         metered_fare,
                "final_price_aed":          final_price,
                "price_per_km_aed":         round(final_price / dist_km, 2),

                # ── Outcome
                "booking_status":           outcome,
                "cancellation_reason":      reason,
                "captain_rating":           cap_rating,
                "customer_rating":          cust_rating,
                "eta_deviation_min":        eta_dev,
            })
            ride_id += 1

        if d.day == 1:
            elapsed = time.time() - t0
            print(f"  ✓ {d.strftime('%B %Y'):12s} — {len(rows):>7,} records  [{elapsed:.1f}s]")

    # ── Save ─────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    elapsed = time.time() - t0
    print(f"\n{'=' * 65}")
    print(f"  ✓ Saved: {out}")
    print(f"  ✓ Records : {len(df):,}")
    print(f"  ✓ Features : {len(df.columns)}")
    print(f"  ✓ Time     : {elapsed:.1f}s")
    print(f"{'=' * 65}")
    print(f"\n  KEY STATISTICS")
    print(f"  Completion rate  : {(df['booking_status'] == 'Completed').mean():.1%}")
    print(f"  Avg final price  : AED {df['final_price_aed'].mean():.2f}")
    print(f"  Avg surge mult   : {df['surge_multiplier'].mean():.3f}")
    print(f"  Max surge mult   : {df['surge_multiplier'].max():.2f}x")
    print(f"  Avg wait time    : {df['wait_time_min'].mean():.1f} min")
    print(f"  Avg distance     : {df['route_distance_km'].mean():.1f} km")
    print(f"  Rain days        : {df[df['is_rain']]['date'].nunique()} days")
    print(f"  Sandstorm days   : {df[df['is_sandstorm']]['date'].nunique()} days")
    print(f"  Ramadan rides    : {df['is_ramadan'].sum():,}")
    print(f"  Airport rides    : {df['is_airport_ride'].sum():,}")
    print(f"  Careem Plus rides: {df['is_careem_plus'].sum():,}")

    print(f"\n  BOOKING STATUS")
    print(df["booking_status"].value_counts().to_string())

    print(f"\n  TOP 5 EVENTS BY RIDES")
    ev = df[df["active_event"] != "None"]["active_event"].value_counts().head(5)
    print(ev.to_string())

    print(f"\n  AVG PRICE BY PRODUCT (AED)")
    pp = df[df["booking_status"] == "Completed"].groupby("product_type")["final_price_aed"].mean().round(2)
    print(pp.to_string())

    return df


# ═════════════════════════════════════════════════════════════════════════════
# 9. AUTO-GENERATE DATA DICTIONARY
# ═════════════════════════════════════════════════════════════════════════════
DATA_DICTIONARY = """# Data Dictionary — Dubai Ride-Hailing Mirror Dataset 2025
## XPrice Project | MIT 622 Final Project | Group 1

Generated by: `data/generate_dataset.py`
Records: ~160,000 | Features: 48 | Period: Jan 1 – Dec 31, 2025 | City: Dubai, UAE

---

## IDENTIFIERS
| Column | Type | Description |
|--------|------|-------------|
| ride_id | str | Unique ride identifier (DXB-2025-XXXXXXX) |
| customer_id | str | Anonymized customer identifier |
| captain_id | str | Anonymized captain (driver) identifier |

## TEMPORAL FEATURES
| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | Full booking datetime (YYYY-MM-DD HH:MM:SS) |
| date | date | Booking date |
| hour | int | Hour of booking (0–23) |
| minute | int | Minute of booking (0–59) |
| day_of_week | int | Day of week (0=Mon … 6=Sun) |
| day_name | str | Day name (Monday … Sunday) |
| week_of_year | int | ISO week number (1–53) |
| month | int | Month (1–12) |
| month_name | str | Month name |
| quarter | int | Quarter (1–4) |
| is_weekend | bool | True if Friday or Saturday (UAE weekend) |
| is_peak_hour | bool | True if hour in {7,8,9,17,18,19,20} |
| is_late_night | bool | True if hour in {0,1,2,3,4} |
| is_ramadan | bool | True if date falls in Ramadan 2025 (Mar 1–29) |
| is_uae_public_holiday | bool | True if UAE official public holiday |
| is_suhoor_window | bool | Ramadan pre-dawn meal window (Ramadan, 00:00–03:00) |
| is_iftar_window | bool | Ramadan sunset meal window (Ramadan, 17:00–18:00) — highest surge period |

## EVENT FEATURES
| Column | Type | Description |
|--------|------|-------------|
| active_event | str | Name of active major event, or "None" |
| event_type | str | Event category (festival, conference, public_holiday, etc.) |
| event_demand_multiplier | float | Demand boost factor from event (1.0 = no event) |

## WEATHER FEATURES
| Column | Type | Description |
|--------|------|-------------|
| temperature_c | float | Air temperature in °C (Dubai range: 14–48°C) |
| humidity_pct | float | Relative humidity % |
| is_rain | bool | True if it rained that day (rare in Dubai — ~8 days/year) |
| is_sandstorm | bool | True if sandstorm conditions |
| weather_demand_factor | float | Demand multiplier from weather (1.0 = clear; up to 1.85 for rain) |

## GEOGRAPHIC FEATURES
| Column | Type | Description |
|--------|------|-------------|
| pickup_zone | str | Pickup zone name (16 Dubai zones) |
| dropoff_zone | str | Dropoff zone name |
| pickup_lat | float | Pickup latitude (jittered within zone centroid ±0.02°) |
| pickup_lon | float | Pickup longitude |
| dropoff_lat | float | Dropoff latitude |
| dropoff_lon | float | Dropoff longitude |
| pickup_area_type | str | Zone type (tourist_commercial, airport, industrial, etc.) |
| dropoff_area_type | str | Zone type of dropoff |
| is_airport_ride | bool | True if pickup or dropoff is Dubai Airport (DXB) |
| is_intrazone_trip | bool | True if pickup and dropoff are in same zone |

## TRIP FEATURES
| Column | Type | Description |
|--------|------|-------------|
| product_type | str | Careem product (Go, Go+, Business, Hala Taxi, Hala EV) |
| route_distance_km | float | Road distance in km (calibrated to Google Maps routing) |
| trip_duration_min | float | Estimated trip duration in minutes |
| avg_speed_kmh | float | Implied average speed (distance/duration) |
| payment_method | str | Payment type (Card, Cash, Careem Pay, Careem Plus) |
| is_careem_plus | bool | True if paid via Careem Plus subscription |

## SUPPLY FEATURES
| Column | Type | Description |
|--------|------|-------------|
| captain_availability_score | float | Captain supply score for pickup zone at booking time (0–1; 1 = fully available) |
| wait_time_min | float | Estimated time to captain arrival (VTAT) in minutes |

## PRICING FEATURES (ML TARGET)
| Column | Type | Description |
|--------|------|-------------|
| surge_multiplier | float | Surge multiplier applied (1.0–2.5x; 1.0 = no surge) |
| metered_fare_aed | float | Base metered fare before surge (AED) |
| final_price_aed | float | **PRIMARY TARGET** — Final fare paid by customer (AED) |
| price_per_km_aed | float | Derived: final_price / route_distance |

## OUTCOME FEATURES
| Column | Type | Description |
|--------|------|-------------|
| booking_status | str | Completed / Captain Cancelled / Customer Cancelled / No Captain Found |
| cancellation_reason | str | Reason for cancellation (N/A for completed rides) |
| captain_rating | float | Captain's rating of customer (1–5; null if not completed) |
| customer_rating | float | Customer's rating of captain (1–5; null if not completed) |
| eta_deviation_min | float | ETA accuracy: actual_arrival − promised_ETA in minutes (null if not completed) |

---

## ZONE REFERENCE TABLE
| Zone | Lat | Lon | Area Type | Demand Tier |
|------|-----|-----|-----------|-------------|
| Downtown Dubai | 25.1972 | 55.2796 | tourist_commercial | very_high |
| Business Bay | 25.1865 | 55.2628 | commercial | high |
| DIFC | 25.2118 | 55.2826 | financial_commercial | high |
| Dubai Marina | 25.0763 | 55.1303 | tourist_residential | very_high |
| JBR | 25.0774 | 55.1296 | tourist_beach | high |
| Palm Jumeirah | 25.1124 | 55.1390 | luxury_residential | high |
| Jumeirah | 25.2084 | 55.2450 | residential | medium |
| Deira | 25.2631 | 55.3246 | commercial_old_city | high |
| Bur Dubai | 25.2532 | 55.3000 | mixed_old_city | medium |
| Dubai Airport (DXB) | 25.2532 | 55.3657 | airport | very_high |
| Al Quoz | 25.1548 | 55.2347 | industrial | low |
| Dubai Hills | 25.0912 | 55.2400 | residential_suburban | medium |
| Al Barsha | 25.1059 | 55.2015 | residential_commercial | medium |
| Mirdif | 25.2197 | 55.4144 | residential_suburban | medium |
| Karama | 25.2375 | 55.3057 | residential_commercial | medium |
| Sharjah Border | 25.3463 | 55.4213 | border_transit | low |
"""


if __name__ == "__main__":
    # Write data dictionary
    dict_path = "data/DATA_DICTIONARY.md"
    os.makedirs("data", exist_ok=True)
    with open(dict_path, "w", encoding="utf-8") as f:
        f.write(DATA_DICTIONARY)
    print(f"Data dictionary written to {dict_path}\n")

    # Generate dataset
    df = generate(target=160_000, out="data/processed/dubai_rides_2025.csv")
