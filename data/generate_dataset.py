"""
XPrice — Dubai Ride-Hailing Mirror Dataset Generator v3.0
==========================================================
Generates ~165,000 realistic Dubai ride records for 2025.

Key improvements over v2:
  • 20 pricing zones derived from official Dubai GeoJSON boundaries
  • Real road distances from OSRM zone-centroid matrix (no more hardcoded values)
  • Location-aware event demand: multiplier decays with distance from event venue
  • Unified traffic / captain-availability formulas (shared with app inference)
  • Careem Plus 5% loyalty discount applied to private-hire fares
  • Monthly fuel-index variation on per-km rate (±AED 0.06 as per RTA policy)
  • Realistic pricing noise (~2.5%) so R² stays meaningful (~0.85-0.90)
  • Salik gate counts from zone_config.py (same as app)

Pricing model calibrated to RTA November 2025 dynamic fare structure.
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from zone_config import (
    ZONE_META, ZONE_NAMES, SALIK,
    get_zone_for_point, get_salik,
)

OUT_PATH  = os.path.join(BASE_DIR, "data", "processed", "dubai_rides_2025.csv")
DIST_PATH = os.path.join(BASE_DIR, "data", "processed", "zone_distances.json")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

N_RIDES = 165_000

# ── Zone arrays ───────────────────────────────────────────────────────────────
Z_LAT  = np.array([ZONE_META[z]["lat"]   for z in ZONE_NAMES])
Z_LON  = np.array([ZONE_META[z]["lon"]   for z in ZONE_NAMES])
Z_TIER = [ZONE_META[z]["tier"]  for z in ZONE_NAMES]
Z_DM   = np.array([ZONE_META[z]["dmult"] for z in ZONE_NAMES])
N_ZONES = len(ZONE_NAMES)
ZONE_IDX = {z: i for i, z in enumerate(ZONE_NAMES)}

# ── Load OSRM distance matrix ─────────────────────────────────────────────────
with open(DIST_PATH) as f:
    _DIST_JSON = json.load(f)

def _osrm_dist(a: str, b: str) -> float:
    key = f"{a}|{b}"
    rev = f"{b}|{a}"
    entry = _DIST_JSON.get(key) or _DIST_JSON.get(rev)
    if entry:
        return float(entry["distance_km"])
    # graceful fallback: haversine × 1.35
    la, loa = ZONE_META[a]["lat"], ZONE_META[a]["lon"]
    lb, lob = ZONE_META[b]["lat"], ZONE_META[b]["lon"]
    dlat = np.deg2rad(lb - la); dlon = np.deg2rad(lob - loa)
    aa = np.sin(dlat/2)**2 + np.cos(np.deg2rad(la))*np.cos(np.deg2rad(lb))*np.sin(dlon/2)**2
    return float(6371 * 2 * np.arctan2(np.sqrt(aa), np.sqrt(1-aa))) * 1.35

def _osrm_dur(a: str, b: str) -> float:
    key = f"{a}|{b}"
    rev = f"{b}|{a}"
    entry = _DIST_JSON.get(key) or _DIST_JSON.get(rev)
    return float(entry["duration_min"]) if entry else None


# ── Precompute zone-pair matrices ─────────────────────────────────────────────
DIST_MATRIX_NP  = np.zeros((N_ZONES, N_ZONES))
DUR_MATRIX_NP   = np.zeros((N_ZONES, N_ZONES))
SALIK_MATRIX_NP = np.zeros((N_ZONES, N_ZONES), dtype=int)
for i, a in enumerate(ZONE_NAMES):
    for j, b in enumerate(ZONE_NAMES):
        if i != j:
            DIST_MATRIX_NP[i, j] = _osrm_dist(a, b)
            d = _osrm_dur(a, b)
            DUR_MATRIX_NP[i, j]  = d if d else (DIST_MATRIX_NP[i, j] / 40.0 * 60)
            SALIK_MATRIX_NP[i, j] = get_salik(a, b)


def get_dist_arr(pu_idx: np.ndarray, do_idx: np.ndarray) -> np.ndarray:
    """Vectorised: zone-pair road distance + within-zone scatter."""
    base = DIST_MATRIX_NP[pu_idx, do_idx]
    intra = (pu_idx == do_idx)
    noise = np.where(
        intra,
        np.random.uniform(1.2, 4.8, len(pu_idx)),
        base * np.random.uniform(0.93, 1.08, len(pu_idx)),
    )
    return np.maximum(noise, 0.5)


def haversine_arr(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1 = np.deg2rad(lat1); p2 = np.deg2rad(lat2)
    dp = np.deg2rad(lat2 - lat1); dl = np.deg2rad(lon2 - lon1)
    a  = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return r * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def bearing_arr(lat1, lon1, lat2, lon2):
    p1 = np.deg2rad(lat1); p2 = np.deg2rad(lat2)
    dl = np.deg2rad(lon2 - lon1)
    y = np.sin(dl)*np.cos(p2)
    x = np.cos(p1)*np.sin(p2) - np.sin(p1)*np.cos(p2)*np.cos(dl)
    return (np.rad2deg(np.arctan2(y, x)) + 360) % 360


# ── Events calendar 2025 with venue zone ─────────────────────────────────────
EVENTS = [
    {"name": "Dubai Shopping Festival", "type": "Shopping Festival",
     "start": "2025-01-03", "end": "2025-02-01",  "dmult": 1.35, "venue_zone": "Deira"},
    {"name": "Dubai Food Festival",     "type": "Food Festival",
     "start": "2025-02-20", "end": "2025-03-08",  "dmult": 1.20, "venue_zone": "Jumeirah"},
    {"name": "Art Dubai",               "type": "Art/Culture",
     "start": "2025-03-18", "end": "2025-03-23",  "dmult": 1.25, "venue_zone": "DIFC"},
    {"name": "Dubai World Cup",         "type": "Sports Event",
     "start": "2025-03-29", "end": "2025-03-29",  "dmult": 1.55, "venue_zone": "Mirdif"},
    {"name": "Eid Al Fitr",             "type": "Religious Holiday",
     "start": "2025-03-30", "end": "2025-04-02",  "dmult": 1.30, "venue_zone": None},
    {"name": "Formula 1 Weekend",       "type": "Sports Event",
     "start": "2025-04-04", "end": "2025-04-06",  "dmult": 1.50, "venue_zone": "Yas Island"},
    {"name": "Eid Al Adha",             "type": "Religious Holiday",
     "start": "2025-06-05", "end": "2025-06-09",  "dmult": 1.25, "venue_zone": None},
    {"name": "GITEX Global",            "type": "Tech Conference",
     "start": "2025-10-13", "end": "2025-10-17",  "dmult": 1.45, "venue_zone": "DIFC"},
    {"name": "Diwali",                  "type": "Cultural Event",
     "start": "2025-10-20", "end": "2025-10-21",  "dmult": 1.20, "venue_zone": "Bur Dubai"},
    {"name": "Dubai Airshow",           "type": "Trade Show",
     "start": "2025-11-17", "end": "2025-11-21",  "dmult": 1.40, "venue_zone": "Dubai South"},
    {"name": "UAE National Day",        "type": "National Holiday",
     "start": "2025-12-02", "end": "2025-12-03",  "dmult": 1.35, "venue_zone": None},
    {"name": "NYE Burj Khalifa",        "type": "New Year Event",
     "start": "2025-12-31", "end": "2025-12-31",  "dmult": 2.20, "venue_zone": "Downtown"},
]
RAMADAN_START = pd.Timestamp("2025-03-01")
RAMADAN_END   = pd.Timestamp("2025-03-29")
UAE_HOLIDAYS = {
    "2025-01-01","2025-03-30","2025-03-31","2025-04-01","2025-04-02","2025-04-03",
    "2025-06-05","2025-06-06","2025-06-07","2025-06-08","2025-06-09",
    "2025-06-26","2025-09-04","2025-12-02","2025-12-03",
}

# Precompute zone-pair haversine distance matrix (for event decay)
_ZONE_HAVERSINE = np.zeros((N_ZONES, N_ZONES))
for i, a in enumerate(ZONE_NAMES):
    for j, b in enumerate(ZONE_NAMES):
        _ZONE_HAVERSINE[i, j] = haversine_arr(
            ZONE_META[a]["lat"], ZONE_META[a]["lon"],
            ZONE_META[b]["lat"], ZONE_META[b]["lon"],
        )


def event_multiplier_for_zone(ev: dict, zone: str) -> float:
    """Location-aware: multiplier decays with km from event venue."""
    venue = ev.get("venue_zone")
    if venue is None:
        return ev["dmult"]   # city-wide (Eid, National Day)
    if venue not in ZONE_IDX:
        return 1.0 + 0.3 * (ev["dmult"] - 1.0)  # off-map venue: mild city effect
    vi = ZONE_IDX[venue]
    zi = ZONE_IDX.get(zone)
    if zi is None:
        return 1.0
    dist = float(_ZONE_HAVERSINE[vi, zi])
    if dist < 3:
        decay = 1.00
    elif dist < 8:
        decay = 0.75
    elif dist < 18:
        decay = 0.40
    else:
        decay = 0.10
    return 1.0 + decay * (ev["dmult"] - 1.0)


# ── Products ──────────────────────────────────────────────────────────────────
PROD_DEF = {
    "Comfort":      (5.00, 5.50, 2.50, 0.40, 15, 5.00, 3.00, 2.50, 0, 0.18, 0.05),
    "Executive":    (5.00, 5.50, 3.20, 0.55, 18, 6.00, 4.00, 3.50, 0, 0.10, 0.08),
    "Hala Taxi":    (5.00, 5.50, 2.20, 0.50, 13, 7.50, 4.00, 4.00, 1, 0.30, 0.25),
    "Eco Friendly": (5.00, 5.50, 2.60, 0.42, 15, 5.00, 3.50, 3.00, 0, 0.08, 0.04),
    "Electric":     (5.00, 5.50, 2.80, 0.45, 16, 5.50, 4.00, 3.50, 0, 0.05, 0.03),
    "Kids":         (5.00, 5.50, 2.80, 0.45, 18, 6.00, 4.50, 3.50, 0, 0.04, 0.02),
    "Hala Kids":    (5.00, 5.50, 2.30, 0.50, 15, 7.50, 4.50, 4.00, 1, 0.03, 0.03),
    "Premier":      (5.00, 5.50, 4.50, 0.80, 30, 8.00, 6.00, 5.00, 0, 0.04, 0.08),
    "MAX":          (5.00, 5.50, 3.80, 0.65, 25, 7.00, 5.00, 4.50, 0, 0.05, 0.22),
    "Hala MAX":     (5.00, 5.50, 2.80, 0.60, 20, 8.00, 5.50, 5.00, 1, 0.13, 0.20),
}
PROD_NAMES = list(PROD_DEF.keys())
PD = np.array([list(v) for v in PROD_DEF.values()])
I_BDAY, I_BNGT, I_PKM, I_PMIN, I_MINF = 0, 1, 2, 3, 4
I_BKPK, I_BKOF, I_BKNT, I_HALA        = 5, 6, 7, 8
I_SHREG, I_SHAPT                        = 9, 10
SH_REG = PD[:, I_SHREG] / PD[:, I_SHREG].sum()
SH_APT = PD[:, I_SHAPT] / PD[:, I_SHAPT].sum()

# Monthly fuel-index variation on per-km rate (±0.06 AED, as per RTA policy)
FUEL_INDEX = np.array([
    0.00, -0.01, -0.02, -0.03, +0.01, +0.04,
    +0.06, +0.05, +0.03, +0.00, -0.02, -0.01,
])   # index by month-1

# ── Demand profiles ───────────────────────────────────────────────────────────
H_REG = np.array([
    0.30, 0.18, 0.12, 0.10, 0.12, 0.22,
    0.55, 0.90, 1.00, 0.85, 0.68, 0.62,
    0.65, 0.63, 0.68, 0.72, 0.90, 0.95,
    1.00, 0.92, 0.82, 0.72, 0.60, 0.45,
])
H_RAM = np.array([
    0.65, 0.60, 0.55, 0.40, 0.20, 0.15,
    0.18, 0.22, 0.25, 0.25, 0.25, 0.25,
    0.28, 0.28, 0.30, 0.35, 0.50, 1.00,
    0.95, 0.85, 0.80, 0.78, 0.80, 0.78,
])
H_REG /= H_REG.sum(); H_RAM /= H_RAM.sum()

MONTH_VOL = np.array([1.10, 1.05, 1.00, 0.90, 0.82, 0.72,
                       0.70, 0.72, 0.85, 1.00, 1.10, 1.12])

# ── Weather by month ──────────────────────────────────────────────────────────
WEATHER = {
    1:  {"temp":(20,25),"hum":(60,75),"rain_p":0.045,"storm_p":0.010},
    2:  {"temp":(21,27),"hum":(55,72),"rain_p":0.035,"storm_p":0.008},
    3:  {"temp":(24,31),"hum":(50,68),"rain_p":0.030,"storm_p":0.012},
    4:  {"temp":(28,35),"hum":(40,60),"rain_p":0.020,"storm_p":0.018},
    5:  {"temp":(33,39),"hum":(40,58),"rain_p":0.005,"storm_p":0.020},
    6:  {"temp":(35,42),"hum":(50,70),"rain_p":0.002,"storm_p":0.025},
    7:  {"temp":(36,43),"hum":(55,78),"rain_p":0.002,"storm_p":0.020},
    8:  {"temp":(36,42),"hum":(55,80),"rain_p":0.002,"storm_p":0.015},
    9:  {"temp":(32,38),"hum":(55,75),"rain_p":0.005,"storm_p":0.012},
    10: {"temp":(28,34),"hum":(50,68),"rain_p":0.015,"storm_p":0.010},
    11: {"temp":(23,30),"hum":(55,72),"rain_p":0.030,"storm_p":0.008},
    12: {"temp":(19,25),"hum":(55,72),"rain_p":0.040,"storm_p":0.008},
}
PAY_METHODS = ["Credit Card","Cash","Careem Pay","Careem Plus"]
PAY_PROBS   = [0.42, 0.30, 0.18, 0.10]

# Zone pickup weights (higher = more pickup demand)
ZONE_PICK_W = np.array([
    1.80,  # Downtown
    1.50,  # Business Bay
    1.40,  # DIFC
    1.20,  # Bur Dubai
    1.20,  # Deira
    1.10,  # Jumeirah
    1.00,  # Al Barsha
    0.70,  # Al Quoz
    1.60,  # Marina
    1.10,  # Palm Jumeirah
    1.20,  # DXB Airport
    0.80,  # Al Nahda
    0.85,  # Mirdif
    0.65,  # Silicon Oasis
    0.90,  # Dubai Hills
    0.75,  # JVC
    0.50,  # Jebel Ali
    0.35,  # Dubai South
    0.55,  # Ras Al Khor
    0.70,  # Sharjah
])
ZONE_PICK_W_APT = ZONE_PICK_W.copy()
ZONE_PICK_W_APT[ZONE_NAMES.index("DXB Airport")] *= 2.0
ZONE_PICK_W_APT /= ZONE_PICK_W_APT.sum()
ZONE_PICK_W     /= ZONE_PICK_W.sum()

# ── 1. Timestamps ─────────────────────────────────────────────────────────────
print("Sampling timestamps...")
year_start  = pd.Timestamp("2025-01-01")
month_ends  = [31,59,90,120,151,181,212,243,273,304,334,365]
day_weights = np.zeros(365)
for m in range(12):
    ds = 0 if m==0 else month_ends[m-1]
    day_weights[ds:month_ends[m]] = MONTH_VOL[m]
day_weights /= day_weights.sum()

day_of_year  = np.random.choice(365, size=N_RIDES, p=day_weights)
timestamps_d = year_start + pd.to_timedelta(day_of_year, unit="D")
months       = timestamps_d.month.values

is_ram_day = (timestamps_d >= RAMADAN_START) & (timestamps_d <= RAMADAN_END)
hour_choice = np.zeros(N_RIDES, dtype=int)
ram_idx = np.where(is_ram_day)[0]; reg_idx = np.where(~is_ram_day)[0]
hour_choice[ram_idx] = np.random.choice(24, size=len(ram_idx), p=H_RAM)
hour_choice[reg_idx] = np.random.choice(24, size=len(reg_idx), p=H_REG)
minutes    = np.random.randint(0, 60, N_RIDES)
timestamps = (
    timestamps_d
    + pd.to_timedelta(hour_choice.astype(int), unit="h")
    + pd.to_timedelta(minutes, unit="m")
)

# ── 2. Temporal flags ─────────────────────────────────────────────────────────
dow        = timestamps.dayofweek.values
# UAE work week: Mon–Fri (0–4). Weekend: Saturday=5, Sunday=6
is_sat     = (dow == 5); is_sun = (dow == 6)
is_weekend = is_sat | is_sun
is_peak    = ((hour_choice >= 8) & (hour_choice < 10)) | ((hour_choice >= 16) & (hour_choice < 20))
is_night   = (hour_choice >= 22) | (hour_choice < 6)
is_offpk   = (~is_peak) & (~is_night)
is_ramadan = np.asarray(is_ram_day)
is_suhoor  = is_ramadan & ((hour_choice >= 1) & (hour_choice <= 3))
is_iftar   = is_ramadan & (hour_choice == 17)
date_strs  = timestamps.strftime("%Y-%m-%d")
is_holiday = np.array([d in UAE_HOLIDAYS for d in date_strs])

# ── 3. Events (location-aware) ────────────────────────────────────────────────
print("Computing location-aware event multipliers...")
active_event = np.full(N_RIDES, "None", dtype=object)
event_type   = np.full(N_RIDES, "None", dtype=object)
event_dmult  = np.ones(N_RIDES)

# ── 4. Zones ──────────────────────────────────────────────────────────────────
print("Sampling zones...")
pu_idx = np.random.choice(N_ZONES, size=N_RIDES, p=ZONE_PICK_W_APT)
COND_DO = np.zeros((N_ZONES, N_ZONES))
for _p in range(N_ZONES):
    _pr = ZONE_PICK_W.copy(); _pr[_p] = 0.0; _pr /= _pr.sum()
    COND_DO[_p] = _pr
do_idx = np.zeros(N_RIDES, dtype=int)
for _p in range(N_ZONES):
    _m = pu_idx == _p
    if _m.sum():
        do_idx[_m] = np.random.choice(N_ZONES, size=_m.sum(), p=COND_DO[_p])

pickup_zone  = np.array(ZONE_NAMES)[pu_idx]
dropoff_zone = np.array(ZONE_NAMES)[do_idx]
is_airport   = (pickup_zone == "DXB Airport") | (dropoff_zone == "DXB Airport")
is_intrazone = (pu_idx == do_idx)

# Apply event multipliers per pickup zone
for ev in EVENTS:
    mask = (timestamps_d >= pd.Timestamp(ev["start"])) & (timestamps_d <= pd.Timestamp(ev["end"]))
    active_event[mask] = ev["name"]
    event_type[mask]   = ev["type"]
    # Compute per-ride multiplier based on pickup zone distance from venue
    for z_name in ZONE_NAMES:
        z_mask = mask & (pickup_zone == z_name)
        if z_mask.sum():
            event_dmult[z_mask] = np.maximum(
                event_dmult[z_mask],
                event_multiplier_for_zone(ev, z_name),
            )

# Jitter coordinates within zone (realistic scatter)
pickup_lat  = Z_LAT[pu_idx] + np.random.normal(0, 0.012, N_RIDES)
pickup_lon  = Z_LON[pu_idx] + np.random.normal(0, 0.012, N_RIDES)
dropoff_lat = Z_LAT[do_idx] + np.random.normal(0, 0.012, N_RIDES)
dropoff_lon = Z_LON[do_idx] + np.random.normal(0, 0.012, N_RIDES)

# ── 5. Distances and Salik ────────────────────────────────────────────────────
print("Computing distances and Salik gates...")
route_direct_distance_km = haversine_arr(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
route_distance_seed      = get_dist_arr(pu_idx, do_idx)
route_distance_km        = np.maximum(route_distance_seed, route_direct_distance_km * np.random.uniform(1.05, 1.35, N_RIDES))
route_efficiency_ratio   = route_distance_km / np.maximum(route_direct_distance_km, 0.5)
route_bearing_deg        = bearing_arr(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
salik_gates              = SALIK_MATRIX_NP[pu_idx, do_idx]

# ── 6. Weather ────────────────────────────────────────────────────────────────
temperature_c = np.zeros(N_RIDES)
humidity_pct  = np.zeros(N_RIDES)
is_rain       = np.zeros(N_RIDES, dtype=bool)
is_sandstorm  = np.zeros(N_RIDES, dtype=bool)
for m in range(1,13):
    mask = months == m; w = WEATHER[m]; n = mask.sum()
    temperature_c[mask] = np.random.uniform(*w["temp"], n)
    humidity_pct[mask]  = np.random.uniform(*w["hum"],  n)
    is_rain[mask]       = np.random.random(n) < w["rain_p"]
    is_sandstorm[mask]  = np.random.random(n) < w["storm_p"]
weather_dmult = 1.0 + 0.40*is_rain.astype(float) + 0.25*is_sandstorm.astype(float)

# ── 7. Products ───────────────────────────────────────────────────────────────
print("Assigning products...")
prod_idx = np.where(
    is_airport,
    np.random.choice(len(PROD_NAMES), size=N_RIDES, p=SH_APT),
    np.random.choice(len(PROD_NAMES), size=N_RIDES, p=SH_REG)
)
product_type     = np.array(PROD_NAMES)[prod_idx]
is_hala_product  = PD[prod_idx, I_HALA].astype(bool)
payment_method   = np.random.choice(PAY_METHODS, size=N_RIDES, p=PAY_PROBS)
is_careem_plus   = (payment_method == "Careem Plus")

# ── 8. Demand / supply / traffic (UNIFIED formula) ───────────────────────────
temporal_demand = np.where(is_peak, 1.18, np.where(is_night, 0.92, 1.00))
ramadan_demand  = np.where(is_iftar, 1.35, np.where(is_suhoor, 1.15, np.where(is_ramadan, 0.96, 1.00)))
weekend_demand  = np.where(is_weekend, 1.05, 1.00)
demand_index    = np.clip(
    Z_DM[pu_idx] * event_dmult * weather_dmult * temporal_demand * ramadan_demand * weekend_demand,
    0.75, 3.00,
)

captain_avail = np.clip(
    1.0 - 0.32*(demand_index - 1.0) + np.random.normal(0, 0.08, N_RIDES),
    0.15, 1.00
)
supply_pressure_index = np.clip(1.0 - captain_avail, 0.0, 1.0)

# ── Unified traffic index formula (matches app inference) ────────────────────
traffic_index = np.clip(
    0.82
    + 0.32 * is_peak.astype(float)
    + 0.10 * is_weekend.astype(float)
    + 0.14 * (event_dmult - 1.0)
    + 0.10 * (weather_dmult - 1.0)
    + 0.08 * is_airport.astype(float)
    + 0.06 * np.clip(route_efficiency_ratio - 1.0, 0.0, 1.5)
    + np.random.normal(0, 0.04, N_RIDES),
    0.68, 2.20,
)
free_flow_speed = np.where(is_peak,
    np.random.uniform(38, 52, N_RIDES),
    np.where(is_night,
        np.random.uniform(58, 72, N_RIDES),
        np.random.uniform(44, 58, N_RIDES),
    )
)
avg_speed          = np.clip(free_flow_speed / traffic_index, 12.0, 78.0)
trip_duration_min  = (route_distance_km / avg_speed) * 60

wait_base = np.where(is_airport, 5.5, np.where(is_peak, 4.5, 3.0))
wait_time = np.clip(
    wait_base * (1.5 - captain_avail) * (0.95 + 0.20 * traffic_index)
    + np.random.exponential(1.5, N_RIDES),
    1.0, 25.0
)

# ── 9. Pricing ────────────────────────────────────────────────────────────────
print("Computing fares...")
# Monthly fuel index: small per-km adjustment
fuel_adj = FUEL_INDEX[months - 1]   # shape (N_RIDES,)

per_km_arr  = PD[prod_idx, I_PKM] + fuel_adj
per_min_arr = PD[prod_idx, I_PMIN]
min_fare    = PD[prod_idx, I_MINF]
base_day    = PD[prod_idx, I_BDAY]; base_night = PD[prod_idx, I_BNGT]
book_peak   = PD[prod_idx, I_BKPK]; book_off   = PD[prod_idx, I_BKOF]
book_night  = PD[prod_idx, I_BKNT]

flagfall    = np.where(is_night, base_night, base_day)
booking_fee = np.where(is_peak, book_peak, np.where(is_night, book_night, book_off))
salik_cost  = salik_gates * 4.0

# Hala (RTA regulated meter)
hala_apt_pu = is_hala_product & (pickup_zone == "DXB Airport")
hala_start  = np.where(hala_apt_pu, 25.0, flagfall + booking_fee)
hala_fare   = hala_start + per_km_arr * route_distance_km + wait_time * 0.50 + salik_cost
hala_final  = np.maximum(hala_fare, min_fare)

# Private hire (Careem dynamic)
demand_supply_gap = np.clip(demand_index - captain_avail, 0.0, 1.5)
surge_mult  = np.clip(1.0 + demand_supply_gap * 0.55, 1.00, 2.50)
# Round surge to nearest 0.25x (as real Careem does)
surge_mult  = np.round(surge_mult * 4) / 4

ph_base = flagfall + booking_fee
ph_fare = ph_base + (per_km_arr * route_distance_km + per_min_arr * trip_duration_min) * surge_mult + salik_cost
ph_final = np.maximum(ph_fare, min_fare)

# Careem Plus 5% loyalty discount (private hire only)
ph_final = np.where(is_careem_plus & ~is_hala_product, ph_final * 0.95, ph_final)

final_price  = np.where(is_hala_product, hala_final, ph_final)
metered_fare = np.where(is_hala_product, hala_fare,  ph_fare)
surge_out    = np.where(is_hala_product, 1.0, surge_mult)

# Realistic market noise (~2.5%) — the part a model should NOT be able to explain
price_noise = np.random.normal(0, 0.025, N_RIDES)
final_price = np.round(np.maximum(final_price * (1 + price_noise), min_fare), 2)

# ── 10. Outcomes ──────────────────────────────────────────────────────────────
cancel_prob = np.clip(
    0.08 + (1.0 - captain_avail)*0.10 + is_rain.astype(float)*0.03 + is_sandstorm.astype(float)*0.03,
    0.05, 0.28
)
cancel_mask = np.random.random(N_RIDES) < cancel_prob
cancel_why  = np.random.choice(
    ["Captain Cancelled","Customer Cancelled","No Captain Available"],
    size=N_RIDES, p=[0.40, 0.40, 0.20]
)
status      = np.full(N_RIDES, "Completed", dtype=object)
status[cancel_mask] = cancel_why[cancel_mask]
cancel_reason_col = np.full(N_RIDES, "N/A", dtype=object)
cancel_reason_col[cancel_mask] = cancel_why[cancel_mask]

capt_rating = np.round(np.random.normal(4.55, 0.30, N_RIDES).clip(1,5), 1)
cust_rating  = np.round(np.random.normal(4.62, 0.28, N_RIDES).clip(1,5), 1)
capt_rating[cancel_mask] = np.nan
cust_rating[cancel_mask]  = np.nan
eta_dev = np.round(np.random.normal(0, 1.5, N_RIDES), 1)

# ── 11. Assemble DataFrame ────────────────────────────────────────────────────
print("Assembling DataFrame...")
ride_ids = [f"RID{2025000000+i:09d}" for i in range(N_RIDES)]
cust_ids = [f"CUS{np.random.randint(1000000,5000000):07d}" for _ in range(N_RIDES)]
capt_ids = [f"CAP{np.random.randint(100000, 500000):06d}"  for _ in range(N_RIDES)]

df = pd.DataFrame({
    "ride_id":                    ride_ids,
    "customer_id":                cust_ids,
    "captain_id":                 capt_ids,
    "timestamp":                  timestamps.strftime("%Y-%m-%d %H:%M:%S"),
    "date":                       date_strs,
    "hour":                       hour_choice,
    "minute":                     minutes,
    "day_of_week":                dow,
    "day_name":                   timestamps.day_name().values,
    "week_of_year":               timestamps.isocalendar().week.values,
    "month":                      months,
    "month_name":                 timestamps.month_name().values,
    "quarter":                    timestamps.quarter.values,
    "is_weekend":                 is_weekend,
    "is_peak_hour":               is_peak,
    "is_late_night":              is_night,
    "is_offpeak":                 is_offpk,
    "is_ramadan":                 is_ramadan,
    "is_uae_public_holiday":      is_holiday,
    "is_suhoor_window":           is_suhoor,
    "is_iftar_window":            is_iftar,
    "active_event":               active_event,
    "event_type":                 event_type,
    "event_demand_multiplier":    np.round(event_dmult, 3),
    "temperature_c":              np.round(temperature_c, 1),
    "humidity_pct":               np.round(humidity_pct, 1),
    "is_rain":                    is_rain,
    "is_sandstorm":               is_sandstorm,
    "weather_demand_factor":      np.round(weather_dmult, 3),
    "pickup_zone":                pickup_zone,
    "dropoff_zone":               dropoff_zone,
    "pickup_lat":                 np.round(pickup_lat, 6),
    "pickup_lon":                 np.round(pickup_lon, 6),
    "dropoff_lat":                np.round(dropoff_lat, 6),
    "dropoff_lon":                np.round(dropoff_lon, 6),
    "pickup_area_type":           np.array(Z_TIER)[pu_idx],
    "dropoff_area_type":          np.array(Z_TIER)[do_idx],
    "is_airport_ride":            is_airport,
    "is_intrazone_trip":          is_intrazone,
    "pickup_density_score":       np.round(Z_DM[pu_idx], 3),
    "dropoff_density_score":      np.round(Z_DM[do_idx], 3),
    "route_direct_distance_km":   np.round(route_direct_distance_km, 2),
    "route_distance_km":          np.round(route_distance_km, 2),
    "route_efficiency_ratio":     np.round(route_efficiency_ratio, 3),
    "route_bearing_deg":          np.round(route_bearing_deg, 2),
    "salik_gates":                salik_gates,
    "salik_cost_aed":             np.round(salik_cost, 2),
    "product_type":               product_type,
    "is_hala_product":            is_hala_product,
    "payment_method":             payment_method,
    "is_careem_plus":             is_careem_plus,
    "demand_index":               np.round(demand_index, 3),
    "captain_availability_score": np.round(captain_avail, 3),
    "supply_pressure_index":      np.round(supply_pressure_index, 3),
    "traffic_index":              np.round(traffic_index, 3),
    "wait_time_min":              np.round(wait_time, 1),
    "trip_duration_min":          np.round(trip_duration_min, 1),
    "avg_speed_kmh":              np.round(avg_speed, 1),
    "surge_multiplier":           np.round(surge_out, 3),
    "booking_fee_aed":            np.round(booking_fee, 2),
    "metered_fare_aed":           np.round(metered_fare, 2),
    "final_price_aed":            np.round(final_price, 2),
    "price_per_km_aed":           np.round(final_price / np.maximum(route_distance_km, 0.1), 2),
    "booking_status":             status,
    "cancellation_reason":        cancel_reason_col,
    "captain_rating":             capt_rating,
    "customer_rating":            cust_rating,
    "eta_deviation_min":          eta_dev,
})

# ── 12. Validate ──────────────────────────────────────────────────────────────
completed = df[df["booking_status"] == "Completed"]
apt_n = df["is_airport_ride"].sum()
print("\n── Validation ──────────────────────────────────────────────────────")
print(f"  Total rides:        {len(df):,}")
print(f"  Completed:          {len(completed):,} ({100*len(completed)/len(df):.1f}%)")
print(f"  Airport rides:      {apt_n:,} ({100*apt_n/len(df):.1f}%)")
print(f"  Avg price (compl.): AED {completed['final_price_aed'].mean():.2f}")
print(f"  Price range:        AED {df['final_price_aed'].min():.2f} – {df['final_price_aed'].max():.2f}")
print(f"  Avg direct route:   {df['route_direct_distance_km'].mean():.1f} km")
print(f"  Avg road dist:      {df['route_distance_km'].mean():.1f} km")
print(f"  Avg Salik gates:    {df['salik_gates'].mean():.2f}")
print(f"  Avg demand index:   {df['demand_index'].mean():.2f}")
print(f"  Avg traffic index:  {df['traffic_index'].mean():.2f}")
print(f"  Hala rides:         {df['is_hala_product'].sum():,} ({100*df['is_hala_product'].mean():.1f}%)")
print(f"\n  Zone distribution (top 10 pickups):")
for z, n in df['pickup_zone'].value_counts().head(10).items():
    print(f"    {z:<20} {n:>7,} ({100*n/len(df):.1f}%)")

df.to_csv(OUT_PATH, index=False)
print(f"\n✓ Dataset saved → {OUT_PATH}")
print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
