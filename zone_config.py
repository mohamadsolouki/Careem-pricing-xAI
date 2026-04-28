"""
zone_config.py  –  Shared zone configuration for XPrice Dubai dataset.

Uses the official Dubai community boundary polygons (GeoJSON) for accurate
point-in-polygon zone detection via Shapely.  Falls back to nearest centroid
when a coordinate lies outside all mapped polygons (e.g. Sharjah, offshore).

Exported symbols used by both dataset generator and the Streamlit app:
    ZONE_NAMES      – ordered list of pricing zone names
    ZONE_META       – dict[zone_name -> {tier, dmult, lat, lon}]
    SALIK           – dict[(zone_a, zone_b) -> int]  gate count
    get_location_context(lat, lon) -> dict[str, str]
    get_neighborhood_for_point(lat, lon) -> str
    get_zone_for_point(lat, lon) -> str
    get_salik(pu_zone, do_zone) -> int
"""

from __future__ import annotations

import json
import os
import math
from functools import lru_cache

try:
    from shapely.geometry import Point, shape as shapely_shape
    from shapely.prepared import prep as shapely_prep
    _SHAPELY = True
except ImportError:
    _SHAPELY = False


def _normalize_neighborhood_name(name: str) -> str:
    return " ".join((name or "").upper().strip().split())


@lru_cache(maxsize=256)
def _display_neighborhood_name(name: str) -> str:
    display_name = _normalize_neighborhood_name(name).title()
    return display_name.replace("Int'L", "Int'l")

# ── 1. Pricing zone metadata ──────────────────────────────────────────────────
# centroid lat/lon used when OSRM distance is looked up and as fallback snap
ZONE_META: dict[str, dict] = {
    "Downtown":      {"tier": "High",   "dmult": 1.30, "lat": 25.1972, "lon": 55.2784},
    "Business Bay":  {"tier": "High",   "dmult": 1.25, "lat": 25.1845, "lon": 55.2711},
    "DIFC":          {"tier": "High",   "dmult": 1.35, "lat": 25.2185, "lon": 55.2810},
    "Bur Dubai":     {"tier": "Medium", "dmult": 1.05, "lat": 25.2461, "lon": 55.3020},
    "Deira":         {"tier": "Medium", "dmult": 1.05, "lat": 25.2740, "lon": 55.3265},
    "Jumeirah":      {"tier": "Medium", "dmult": 1.15, "lat": 25.1754, "lon": 55.2218},
    "Al Barsha":     {"tier": "Medium", "dmult": 1.05, "lat": 25.0950, "lon": 55.1821},
    "Al Quoz":       {"tier": "Low",    "dmult": 0.90, "lat": 25.1510, "lon": 55.2265},
    "Marina":        {"tier": "High",   "dmult": 1.25, "lat": 25.0847, "lon": 55.1404},
    "Palm Jumeirah": {"tier": "High",   "dmult": 1.20, "lat": 25.1207, "lon": 55.1286},
    "DXB Airport":   {"tier": "High",   "dmult": 1.10, "lat": 25.2532, "lon": 55.3657},
    "Al Nahda":      {"tier": "Medium", "dmult": 0.92, "lat": 25.2825, "lon": 55.3753},
    "Mirdif":        {"tier": "Medium", "dmult": 1.00, "lat": 25.2230, "lon": 55.4350},
    "Silicon Oasis": {"tier": "Low",    "dmult": 0.88, "lat": 25.1200, "lon": 55.3800},
    "Dubai Hills":   {"tier": "Medium", "dmult": 1.05, "lat": 25.1050, "lon": 55.2560},
    "JVC":           {"tier": "Low",    "dmult": 0.90, "lat": 25.0440, "lon": 55.2450},
    "Jebel Ali":     {"tier": "Low",    "dmult": 0.85, "lat": 25.0400, "lon": 55.1280},
    "Dubai South":   {"tier": "Low",    "dmult": 0.80, "lat": 24.9380, "lon": 55.0900},
    "Ras Al Khor":   {"tier": "Low",    "dmult": 0.95, "lat": 25.1880, "lon": 55.3550},
    "Sharjah":       {"tier": "Low",    "dmult": 0.95, "lat": 25.3463, "lon": 55.4209},
}

ZONE_NAMES: list[str] = list(ZONE_META.keys())

# ── 2. GeoJSON neighborhood → pricing zone mapping ───────────────────────────
NEIGHBORHOOD_TO_ZONE: dict[str, str] = {
    # Downtown
    "BURJ KHALIFA": "Downtown", "ZAA'BEEL FIRST": "Downtown",
    "ZAA'BEEL SECOND": "Downtown", "AL WASL": "Downtown",
    # Business Bay
    "BUSINESS BAY": "Business Bay", "BU KADRA": "Business Bay",
    # DIFC
    "TRADE CENTER FIRST": "DIFC", "TRADE CENTER SECOND": "DIFC",
    # Bur Dubai
    "AL RAFFA": "Bur Dubai", "AL JAFILIYA": "Bur Dubai",
    "MANKHOOL": "Bur Dubai", "AL SOUQ AL KABEER": "Bur Dubai",
    "AL SHINDAGHA": "Bur Dubai", "AL HUDAIBA": "Bur Dubai",
    "UMM HURAIR FIRST": "Bur Dubai", "UMM HURAIR SECOND": "Bur Dubai",
    "AL KARAMA": "Bur Dubai", "OUD METHA": "Bur Dubai",
    "AL HAMRIYA": "Bur Dubai", "MADINAT DUBAI AL MELAHEYAH": "Bur Dubai",
    "RIGGAT AL BUTEEN": "Bur Dubai", "AL KIFAF": "Bur Dubai",
    # Deira
    "ABU HAIL": "Deira", "AL MAMZAR": "Deira", "AL MURAQQABAT": "Deira",
    "AL MURAR": "Deira", "AL MUTEENA": "Deira", "AL RIGGA": "Deira",
    "AL RAS": "Deira", "AL DAGHAYA": "Deira", "CORNICHE DEIRA": "Deira",
    "HOR AL ANZ": "Deira", "HOR AL ANZ EAST": "Deira", "AL BARAHA": "Deira",
    "AL KHABAISI": "Deira", "NAIF": "Deira", "AL SABKHA": "Deira",
    "PORT SAEED": "Deira", "AL WUHEIDA": "Deira", "AL BUTEEN": "Deira",
    "AL HAMRIYA PORT": "Deira", "AL CORNICHE": "Deira",
    "AYAL NASIR": "Deira", "NAKHLAT DEIRA": "Deira",
    # Jumeirah (includes Satwa/Bada which are on the Jumeirah side)
    "JUMEIRA FIRST": "Jumeirah", "JUMEIRA SECOND": "Jumeirah",
    "JUMEIRA THIRD": "Jumeirah", "AL SAFA FIRST": "Jumeirah",
    "AL SAFA SECOND": "Jumeirah", "UMM SUQEIM FIRST": "Jumeirah",
    "UMM SUQEIM SECOND": "Jumeirah", "UMM SUQEIM THIRD": "Jumeirah",
    "AL MANARA": "Jumeirah", "UMM AL SHEIF": "Jumeirah",
    "JUMEIRA BAY": "Jumeirah", "AL SAFOUH FIRST": "Jumeirah",
    "AL SAFOUH SECOND": "Jumeirah", "JUMEIRA ISLAND 2": "Jumeirah",
    "AL SATWA": "Jumeirah", "AL BADA'": "Jumeirah",
    # Al Barsha
    "AL BARSHA NORTH": "Al Barsha", "AL BARSHA SOUTH": "Al Barsha",
    "AL THANYAH FIRST": "Al Barsha", "AL THANYAH SECOND": "Al Barsha",
    "AL THANYAH THIRD": "Al Barsha", "AL THANYAH  FOURTH": "Al Barsha",
    "AL THANYAH FIFTH": "Al Barsha",
    # Al Quoz
    "AL QOUZ": "Al Quoz", "AL QOUZ INDUSTRIAL": "Al Quoz",
    # Marina
    "MARSA DUBAI": "Marina", "WORLD ISLANDS": "Marina",
    # Palm Jumeirah
    "NAKHLAT JUMEIRA": "Palm Jumeirah",
    # DXB Airport
    "DUBAI INT'L AIRPORT": "DXB Airport", "UMM RAMOOL": "DXB Airport",
    "AL GARHOUD": "DXB Airport", "AL TWAR": "DXB Airport",
    # Al Nahda / Qusais
    "AL NAHDA FIRST": "Al Nahda", "AL NAHDA SECOND": "Al Nahda",
    "AL QUSAIS": "Al Nahda", "AL QUSAIS INDUSTRIAL": "Al Nahda",
    "MUHAISNAH": "Al Nahda",
    # Mirdif
    "MIRDIF": "Mirdif", "AL WARQA'A": "Mirdif", "OUD AL MUTEENA": "Mirdif",
    "AL RASHIDIYA": "Mirdif", "AL MIZHAR": "Mirdif",
    "MUSHRAIF": "Mirdif", "AL KHWANEEJ": "Mirdif",
    "GHADEER AL TAIR": "Mirdif", "NADD AL HAMAR": "Mirdif",
    # Silicon Oasis / Warsan
    "WADI AL SAFA 2": "Silicon Oasis", "WADI AL SAFA 3": "Silicon Oasis",
    "WADI AL SAFA 4": "Silicon Oasis", "WADI AL SAFA 5": "Silicon Oasis",
    "WADI AL SAFA 6": "Silicon Oasis", "WADI AL SAFA 7": "Silicon Oasis",
    "AL ROWAIYAH FIRST": "Silicon Oasis", "AL ROWAIYAH SECOND": "Silicon Oasis",
    "AL ROWAIYAH THIRD": "Silicon Oasis", "WARSAN": "Silicon Oasis",
    "AL TTAY": "Silicon Oasis", "ALEYAS": "Silicon Oasis",
    # Dubai Hills
    "HADAEQ SHEIKH MOHAMMED BIN RASHID": "Dubai Hills",
    "NADD HESSA": "Dubai Hills", "NADD AL SHIBA": "Dubai Hills",
    # JVC / Sports City / Al Hebiah
    "AL HEBIAH FIRST": "JVC", "AL HEBIAH SECOND": "JVC",
    "AL HEBIAH THIRD": "JVC", "AL HEBIAH FOURTH": "JVC",
    "AL HEBIAH FIFTH": "JVC", "AL HEBIAH SIXTH": "JVC",
    "ME'AISEM FIRST": "JVC", "ME'AISEM SECOND": "JVC",
    "AL YALAYIS": "JVC", "NADD SHAMMA": "JVC",
    # Jebel Ali
    "JABAL ALI FIRST": "Jebel Ali", "JABAL ALI SECOND": "Jebel Ali",
    "JABAL ALI THIRD": "Jebel Ali", "JABAL ALI INDUSTRIAL": "Jebel Ali",
    "MENA JABAL ALI": "Jebel Ali",
    # Dubai South / DIP
    "MADINAT AL MATAAR": "Dubai South",
    "SAIH SHUAIB 1": "Dubai South", "SAIH SHUAIB 2": "Dubai South",
    "SAIH SHUAIB 3": "Dubai South", "SAIH SHUAIB 4": "Dubai South",
    "DUBAI INVESTMENT PARK FIRST": "Dubai South",
    "DUBAI INVESTMENT PARK SECOND": "Dubai South",
    # Ras Al Khor / Creek
    "RAS AL KHOR": "Ras Al Khor", "RAS AL KHOR INDUSTRIAL": "Ras Al Khor",
    "AL JADAF": "Ras Al Khor", "AL KHEERAN": "Ras Al Khor",
    "AL KHEERAN FIRST": "Ras Al Khor", "AL AWIR FIRST": "Ras Al Khor",
    "AL AWIR SECOND": "Ras Al Khor", "AL ATHBAH": "Ras Al Khor",
}

# ── 3. Salik gate counts per zone pair ───────────────────────────────────────
# Based on actual Dubai Salik gate locations (Al Garhoud Bridge, Al Maktoum Bridge,
# Al Safa, Al Barsha, Jebel Ali on SZR; Al Mamzar on Ittihad Road to Sharjah).
# Table is symmetric: code always normalises to canonical key.
SALIK: dict[tuple[str, str], int] = {
    # ── Outbound from Marina / Palm Jumeirah (2 SZR gates: Al Barsha + Al Safa) ──
    ("Marina", "Downtown"):       2, ("Marina", "DIFC"):           2,
    ("Marina", "Business Bay"):   2, ("Marina", "Jumeirah"):        1,
    ("Marina", "Bur Dubai"):      2, ("Marina", "Deira"):           3,
    ("Marina", "DXB Airport"):    3, ("Marina", "Al Barsha"):       0,
    ("Marina", "Al Quoz"):        1, ("Marina", "Al Nahda"):        3,
    ("Marina", "Mirdif"):         3, ("Marina", "Silicon Oasis"):   2,
    ("Marina", "Dubai Hills"):    1, ("Marina", "JVC"):             0,
    ("Marina", "Jebel Ali"):      0, ("Marina", "Dubai South"):     0,
    ("Marina", "Ras Al Khor"):    2, ("Marina", "Sharjah"):         3,
    ("Palm Jumeirah", "Downtown"):      2, ("Palm Jumeirah", "DIFC"):          2,
    ("Palm Jumeirah", "Business Bay"):  2, ("Palm Jumeirah", "Jumeirah"):      1,
    ("Palm Jumeirah", "Bur Dubai"):     2, ("Palm Jumeirah", "Deira"):         3,
    ("Palm Jumeirah", "DXB Airport"):   3, ("Palm Jumeirah", "Al Barsha"):     0,
    ("Palm Jumeirah", "Al Quoz"):       1, ("Palm Jumeirah", "Dubai Hills"):   1,
    ("Palm Jumeirah", "Ras Al Khor"):   2, ("Palm Jumeirah", "Sharjah"):       3,
    # ── Jumeirah outbound (1 SZR gate: Al Safa) ──────────────────────────────
    ("Jumeirah", "Downtown"):      1, ("Jumeirah", "DIFC"):          1,
    ("Jumeirah", "Business Bay"):  1, ("Jumeirah", "Bur Dubai"):     1,
    ("Jumeirah", "Deira"):         2, ("Jumeirah", "DXB Airport"):   2,
    ("Jumeirah", "Al Nahda"):      2, ("Jumeirah", "Mirdif"):        2,
    ("Jumeirah", "Silicon Oasis"): 1, ("Jumeirah", "Dubai Hills"):   0,
    ("Jumeirah", "Ras Al Khor"):   1, ("Jumeirah", "Sharjah"):       2,
    ("Jumeirah", "Jebel Ali"):     1, ("Jumeirah", "Dubai South"):   1,
    # ── Al Barsha outbound (1 SZR gate: Al Safa) ─────────────────────────────
    ("Al Barsha", "Downtown"):     1, ("Al Barsha", "DIFC"):         1,
    ("Al Barsha", "Business Bay"): 1, ("Al Barsha", "Bur Dubai"):    1,
    ("Al Barsha", "Deira"):        2, ("Al Barsha", "DXB Airport"):  2,
    ("Al Barsha", "Al Nahda"):     2, ("Al Barsha", "Mirdif"):       2,
    ("Al Barsha", "Silicon Oasis"):1, ("Al Barsha", "Dubai Hills"):  0,
    ("Al Barsha", "Ras Al Khor"):  1, ("Al Barsha", "Sharjah"):      2,
    ("Al Barsha", "Jebel Ali"):    1, ("Al Barsha", "Dubai South"):  1,
    # ── Al Quoz outbound (1 SZR gate: Al Safa) ───────────────────────────────
    ("Al Quoz", "Downtown"):     1, ("Al Quoz", "DIFC"):         1,
    ("Al Quoz", "Business Bay"): 1, ("Al Quoz", "Bur Dubai"):    1,
    ("Al Quoz", "Deira"):        2, ("Al Quoz", "DXB Airport"):  2,
    ("Al Quoz", "Al Nahda"):     2, ("Al Quoz", "Mirdif"):       2,
    ("Al Quoz", "Silicon Oasis"):1, ("Al Quoz", "Ras Al Khor"):  1,
    ("Al Quoz", "Sharjah"):      2, ("Al Quoz", "Jebel Ali"):    1,
    ("Al Quoz", "Dubai South"):  1,
    # ── Dubai Hills outbound (crosses Al Safa via SZR) ───────────────────────
    ("Dubai Hills", "Downtown"):     1, ("Dubai Hills", "DIFC"):         1,
    ("Dubai Hills", "Business Bay"): 1, ("Dubai Hills", "Bur Dubai"):    1,
    ("Dubai Hills", "Deira"):        2, ("Dubai Hills", "DXB Airport"):  2,
    ("Dubai Hills", "Al Nahda"):     2, ("Dubai Hills", "Mirdif"):       2,
    ("Dubai Hills", "Marina"):       1, ("Dubai Hills", "Palm Jumeirah"):1,
    ("Dubai Hills", "Silicon Oasis"):1, ("Dubai Hills", "Ras Al Khor"):  1,
    ("Dubai Hills", "Sharjah"):      3, ("Dubai Hills", "Jebel Ali"):    1,
    # ── JVC outbound (0-1 gates, south of Al Barsha gate) ────────────────────
    ("JVC", "Downtown"):     1, ("JVC", "DIFC"):         1,
    ("JVC", "Business Bay"): 1, ("JVC", "Bur Dubai"):    1,
    ("JVC", "Deira"):        2, ("JVC", "DXB Airport"):  2,
    ("JVC", "Al Nahda"):     2, ("JVC", "Mirdif"):       2,
    ("JVC", "Silicon Oasis"):1, ("JVC", "Ras Al Khor"):  1,
    ("JVC", "Sharjah"):      2, ("JVC", "Jebel Ali"):    0,
    ("JVC", "Dubai South"):  0, ("JVC", "Marina"):       0,
    # ── Jebel Ali (passes Jebel Ali gate heading toward Dubai) ───────────────
    ("Jebel Ali", "Downtown"):      2, ("Jebel Ali", "DIFC"):          2,
    ("Jebel Ali", "Business Bay"):  2, ("Jebel Ali", "Bur Dubai"):     2,
    ("Jebel Ali", "Deira"):         3, ("Jebel Ali", "DXB Airport"):   3,
    ("Jebel Ali", "Marina"):        1, ("Jebel Ali", "Jumeirah"):      1,
    ("Jebel Ali", "Al Barsha"):     1, ("Jebel Ali", "Al Quoz"):       1,
    ("Jebel Ali", "Dubai Hills"):   1, ("Jebel Ali", "JVC"):           0,
    ("Jebel Ali", "Dubai South"):   1, ("Jebel Ali", "Sharjah"):       3,
    ("Jebel Ali", "Silicon Oasis"): 2, ("Jebel Ali", "Ras Al Khor"):   2,
    # ── Dubai South (passes Jebel Ali gate + Al Barsha) ──────────────────────
    ("Dubai South", "Downtown"):      2, ("Dubai South", "DIFC"):          2,
    ("Dubai South", "Business Bay"):  2, ("Dubai South", "Bur Dubai"):     2,
    ("Dubai South", "Deira"):         3, ("Dubai South", "DXB Airport"):   3,
    ("Dubai South", "Marina"):        1, ("Dubai South", "Jumeirah"):      1,
    ("Dubai South", "Al Barsha"):     1, ("Dubai South", "Sharjah"):       4,
    ("Dubai South", "Silicon Oasis"): 2, ("Dubai South", "JVC"):           0,
    # ── Deira outbound (creek crossing: 1 gate for Maktoum/Garhoud bridge) ──
    ("Deira", "Bur Dubai"):    1, ("Deira", "Downtown"):     1,
    ("Deira", "DIFC"):         1, ("Deira", "Business Bay"): 1,
    ("Deira", "Marina"):       3, ("Deira", "Jumeirah"):     2,
    ("Deira", "Al Barsha"):    2, ("Deira", "Al Quoz"):      2,
    ("Deira", "Dubai Hills"):  2, ("Deira", "Silicon Oasis"):0,
    ("Deira", "Ras Al Khor"):  0, ("Deira", "JVC"):          2,
    ("Deira", "Jebel Ali"):    3,
    # ── DXB Airport (Garhoud Bridge: 1 gate to central Dubai) ────────────────
    ("DXB Airport", "Downtown"):     1, ("DXB Airport", "DIFC"):         1,
    ("DXB Airport", "Business Bay"): 1, ("DXB Airport", "Bur Dubai"):    1,
    ("DXB Airport", "Marina"):       3, ("DXB Airport", "Jumeirah"):     2,
    ("DXB Airport", "Al Barsha"):    2, ("DXB Airport", "Al Quoz"):      2,
    ("DXB Airport", "Dubai Hills"):  2, ("DXB Airport", "Deira"):        0,
    ("DXB Airport", "Al Nahda"):     0, ("DXB Airport", "Mirdif"):       0,
    ("DXB Airport", "Silicon Oasis"):0, ("DXB Airport", "Ras Al Khor"):  0,
    ("DXB Airport", "JVC"):          2, ("DXB Airport", "Jebel Ali"):    3,
    ("DXB Airport", "Sharjah"):      1,
    # ── Mirdif / Al Rashidiya ─────────────────────────────────────────────────
    ("Mirdif", "Downtown"):     1, ("Mirdif", "DIFC"):         1,
    ("Mirdif", "Business Bay"): 1, ("Mirdif", "Bur Dubai"):    1,
    ("Mirdif", "Marina"):       3, ("Mirdif", "Al Barsha"):    2,
    ("Mirdif", "Al Quoz"):      2, ("Mirdif", "Dubai Hills"):  2,
    ("Mirdif", "Silicon Oasis"):0, ("Mirdif", "Ras Al Khor"):  0,
    ("Mirdif", "Sharjah"):      1, ("Mirdif", "JVC"):          2,
    # ── Al Nahda / Qusais ─────────────────────────────────────────────────────
    ("Al Nahda", "Downtown"):     1, ("Al Nahda", "DIFC"):         1,
    ("Al Nahda", "Business Bay"): 1, ("Al Nahda", "Bur Dubai"):    2,
    ("Al Nahda", "Marina"):       3, ("Al Nahda", "Jumeirah"):     2,
    ("Al Nahda", "Al Barsha"):    2, ("Al Nahda", "Silicon Oasis"):0,
    ("Al Nahda", "Ras Al Khor"):  0, ("Al Nahda", "Sharjah"):      1,
    ("Al Nahda", "JVC"):          2,
    # ── Silicon Oasis ─────────────────────────────────────────────────────────
    ("Silicon Oasis", "Downtown"):     0, ("Silicon Oasis", "DIFC"):         0,
    ("Silicon Oasis", "Business Bay"): 0, ("Silicon Oasis", "Bur Dubai"):    1,
    ("Silicon Oasis", "Marina"):       2, ("Silicon Oasis", "Sharjah"):      1,
    ("Silicon Oasis", "JVC"):          1, ("Silicon Oasis", "Dubai Hills"):  1,
    # ── Ras Al Khor ───────────────────────────────────────────────────────────
    ("Ras Al Khor", "Downtown"):     0, ("Ras Al Khor", "DIFC"):         0,
    ("Ras Al Khor", "Business Bay"): 0, ("Ras Al Khor", "Bur Dubai"):    1,
    ("Ras Al Khor", "Deira"):        0, ("Ras Al Khor", "DXB Airport"):  0,
    ("Ras Al Khor", "Marina"):       2, ("Ras Al Khor", "Sharjah"):      1,
    ("Ras Al Khor", "JVC"):          1,
    # ── Sharjah (Al Mamzar border gate to enter Dubai) ───────────────────────
    ("Sharjah", "Deira"):        1, ("Sharjah", "DXB Airport"):  1,
    ("Sharjah", "Al Nahda"):     1, ("Sharjah", "Mirdif"):       1,
    ("Sharjah", "Bur Dubai"):    2, ("Sharjah", "Downtown"):     2,
    ("Sharjah", "DIFC"):         2, ("Sharjah", "Business Bay"): 2,
    ("Sharjah", "Ras Al Khor"):  1, ("Sharjah", "Silicon Oasis"):1,
    ("Sharjah", "Marina"):       3, ("Sharjah", "Jumeirah"):     3,
    ("Sharjah", "Al Barsha"):    3, ("Sharjah", "Dubai Hills"):  3,
    ("Sharjah", "JVC"):          2, ("Sharjah", "Jebel Ali"):    3,
    ("Sharjah", "Dubai South"):  4,
    # ── Bur Dubai ─────────────────────────────────────────────────────────────
    ("Bur Dubai", "Marina"):       2, ("Bur Dubai", "Jumeirah"):     1,
    ("Bur Dubai", "Al Barsha"):    1, ("Bur Dubai", "Al Quoz"):      1,
    ("Bur Dubai", "Dubai Hills"):  1, ("Bur Dubai", "JVC"):          1,
    ("Bur Dubai", "Jebel Ali"):    2, ("Bur Dubai", "Dubai South"):  2,
    ("Bur Dubai", "Silicon Oasis"):1, ("Bur Dubai", "Ras Al Khor"):  1,
    ("Bur Dubai", "Sharjah"):      2,
    # ── Downtown / Business Bay / DIFC outbound ───────────────────────────────
    ("Downtown", "Marina"):       2, ("Downtown", "Jumeirah"):     1,
    ("Downtown", "Al Barsha"):    1, ("Downtown", "Al Quoz"):      1,
    ("Downtown", "Dubai Hills"):  1, ("Downtown", "JVC"):          1,
    ("Downtown", "Jebel Ali"):    2, ("Downtown", "Dubai South"):  2,
    ("Downtown", "Sharjah"):      2, ("Downtown", "Silicon Oasis"):0,
    ("DIFC", "Marina"):       2, ("DIFC", "Jumeirah"):     1,
    ("DIFC", "Al Barsha"):    1, ("DIFC", "Al Quoz"):      1,
    ("DIFC", "Dubai Hills"):  1, ("DIFC", "JVC"):          1,
    ("DIFC", "Jebel Ali"):    2, ("DIFC", "Dubai South"):  2,
    ("DIFC", "Sharjah"):      2, ("DIFC", "Silicon Oasis"):0,
    ("Business Bay", "Marina"):       2, ("Business Bay", "Jumeirah"):     1,
    ("Business Bay", "Al Barsha"):    1, ("Business Bay", "Al Quoz"):      1,
    ("Business Bay", "Dubai Hills"):  1, ("Business Bay", "JVC"):          1,
    ("Business Bay", "Jebel Ali"):    2, ("Business Bay", "Dubai South"):  2,
    ("Business Bay", "Sharjah"):      2, ("Business Bay", "Silicon Oasis"):0,
}


# ── 4. Load GeoJSON and build Shapely lookup ──────────────────────────────────
_GEOJSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dubai_neighborhoods.geojson")

_polygons: list[tuple] = []   # [(prepared_geom, neighborhood_name, zone_name), ...]

def _load_polygons() -> None:
    global _polygons
    if _polygons:
        return
    if not _SHAPELY:
        return
    if not os.path.exists(_GEOJSON_PATH):
        return
    with open(_GEOJSON_PATH, encoding="utf-8") as fh:
        gj = json.load(fh)
    for feat in gj["features"]:
        name = _normalize_neighborhood_name(feat["properties"].get("name", ""))
        zone = NEIGHBORHOOD_TO_ZONE.get(name)
        if zone is None:
            continue
        try:
            geom = shapely_shape(feat["geometry"])
            _polygons.append((shapely_prep(geom), name, zone))
        except Exception:
            pass


def _nearest_zone_for_point(lat: float, lon: float) -> str:
    best, best_d = "Downtown", float("inf")
    for zname, zmeta in ZONE_META.items():
        d = (zmeta["lat"] - lat) ** 2 + (zmeta["lon"] - lon) ** 2
        if d < best_d:
            best_d = d
            best = zname
    return best


def get_location_context(lat: float, lon: float) -> dict[str, str]:
    """Return neighborhood, pricing zone, and lookup source for a coordinate."""
    _load_polygons()
    if _SHAPELY and _polygons:
        pt = Point(lon, lat)   # Shapely uses (x=lon, y=lat)
        for geom, neighborhood, zone in _polygons:
            if geom.contains(pt) or geom.intersects(pt):
                return {
                    "neighborhood": _display_neighborhood_name(neighborhood),
                    "zone": zone,
                    "source": "polygon",
                }

    fallback_zone = _nearest_zone_for_point(lat, lon)
    return {
        "neighborhood": fallback_zone,
        "zone": fallback_zone,
        "source": "centroid",
    }


def get_neighborhood_for_point(lat: float, lon: float) -> str:
    return get_location_context(lat, lon)["neighborhood"]


def get_zone_for_point(lat: float, lon: float) -> str:
    """Return the pricing zone name for a coordinate."""
    return get_location_context(lat, lon)["zone"]


def get_salik(pickup_zone: str, dropoff_zone: str) -> int:
    """Return number of Salik toll gates for a zone pair."""
    if pickup_zone == dropoff_zone:
        return 0
    key = (pickup_zone, dropoff_zone)
    if key in SALIK:
        return SALIK[key]
    rev = (dropoff_zone, pickup_zone)
    if rev in SALIK:
        return SALIK[rev]
    return 0
