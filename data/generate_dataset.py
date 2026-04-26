"""
XPrice — Dubai Ride-Hailing Mirror Dataset Generator v2.0
=========================================================
Generates ~165,000 realistic Dubai ride records for 2025.

Pricing model calibrated to RTA's November 2025 dynamic fare structure:
  Hala products (RTA-regulated taxis):
    - Flagfall AED 5.00 (peak/day) | AED 5.50 (night)
    - Airport pickup flagfall: AED 25 (replaces flagfall + booking fee)
    - Dynamic booking fee: AED 7.50 (peak) | AED 4.00 (off-peak/night)
    - Per-km: AED 2.20 (adjusts +/-AED 0.06 monthly with fuel index)
    - Waiting (stationary): AED 0.50/min
    - Minimum fare: AED 13 (app-booked)
  Private hire products (Careem-set rates):
    - Base fare + booking fee + per-km + per-min (full trip duration)
    - Careem surge multiplier on demand-supply imbalance
  All products:
    - Salik toll: AED 4.00 per gate crossed
  Peak hours (Mon-Thu): 08:00-10:00 and 16:00-20:00
  Peak hours (Fri):     16:00-24:00
  Night:                22:00-06:00
  Off-peak:             all other hours

Sources:
  RTA dynamic taxi fare update (Gulf News, Nov 2025)
  Careem Engineering Blog - YODA ML platform (2020)
  e& FY2025 Annual Report | WTW 2024 MENA Rideshare Report
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "dubai_rides_2025.csv")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

N_RIDES = 165_000

# ── 1. Dubai zones ──────────────────────────────────────────────────────────
ZONES = {
    "Downtown":      {"lat": 25.1972, "lon": 55.2744, "tier": "High",   "dmult": 1.25},
    "Marina":        {"lat": 25.0805, "lon": 55.1403, "tier": "High",   "dmult": 1.20},
    "JBR":           {"lat": 25.0774, "lon": 55.1302, "tier": "High",   "dmult": 1.15},
    "DIFC":          {"lat": 25.2118, "lon": 55.2797, "tier": "High",   "dmult": 1.30},
    "Deira":         {"lat": 25.2697, "lon": 55.3095, "tier": "Medium", "dmult": 1.05},
    "Bur Dubai":     {"lat": 25.2532, "lon": 55.2956, "tier": "Medium", "dmult": 1.05},
    "Jumeirah":      {"lat": 25.2048, "lon": 55.2434, "tier": "Medium", "dmult": 1.10},
    "Al Quoz":       {"lat": 25.1521, "lon": 55.2270, "tier": "Low",    "dmult": 0.90},
    "Business Bay":  {"lat": 25.1867, "lon": 55.2622, "tier": "High",   "dmult": 1.20},
    "Dubai Hills":   {"lat": 25.1150, "lon": 55.2380, "tier": "Medium", "dmult": 1.00},
    "DXB Airport":   {"lat": 25.2532, "lon": 55.3657, "tier": "High",   "dmult": 1.05},
    "Sharjah":       {"lat": 25.3463, "lon": 55.4209, "tier": "Low",    "dmult": 0.95},
}
ZONE_NAMES = list(ZONES.keys())
Z_LAT  = np.array([ZONES[z]["lat"]   for z in ZONE_NAMES])
Z_LON  = np.array([ZONES[z]["lon"]   for z in ZONE_NAMES])
Z_TIER = [ZONES[z]["tier"]  for z in ZONE_NAMES]
Z_DM   = np.array([ZONES[z]["dmult"] for z in ZONE_NAMES])

# ── 2. Zone-pair road distances (km) ────────────────────────────────────────
DIST_MATRIX = {
    ("Downtown",    "Marina"):       24.0,
    ("Downtown",    "JBR"):          26.0,
    ("Downtown",    "DIFC"):          2.5,
    ("Downtown",    "Deira"):         8.0,
    ("Downtown",    "Bur Dubai"):     5.0,
    ("Downtown",    "Jumeirah"):      9.0,
    ("Downtown",    "Al Quoz"):      10.0,
    ("Downtown",    "Business Bay"):  3.5,
    ("Downtown",    "Dubai Hills"):  18.0,
    ("Downtown",    "DXB Airport"):  14.0,
    ("Downtown",    "Sharjah"):      25.0,
    ("Marina",      "JBR"):           2.5,
    ("Marina",      "DIFC"):         22.0,
    ("Marina",      "Deira"):        34.0,
    ("Marina",      "Bur Dubai"):    30.0,
    ("Marina",      "Jumeirah"):     17.0,
    ("Marina",      "Al Quoz"):      15.0,
    ("Marina",      "Business Bay"): 23.0,
    ("Marina",      "Dubai Hills"):  18.0,
    ("Marina",      "DXB Airport"):  38.0,
    ("Marina",      "Sharjah"):      50.0,
    ("JBR",         "DIFC"):         24.0,
    ("JBR",         "Deira"):        36.0,
    ("JBR",         "Bur Dubai"):    32.0,
    ("JBR",         "Jumeirah"):     17.0,
    ("JBR",         "Al Quoz"):      16.0,
    ("JBR",         "Business Bay"): 25.0,
    ("JBR",         "Dubai Hills"):  20.0,
    ("JBR",         "DXB Airport"):  40.0,
    ("JBR",         "Sharjah"):      52.0,
    ("DIFC",        "Deira"):        10.0,
    ("DIFC",        "Bur Dubai"):     6.0,
    ("DIFC",        "Jumeirah"):     10.0,
    ("DIFC",        "Al Quoz"):      10.0,
    ("DIFC",        "Business Bay"):  2.5,
    ("DIFC",        "Dubai Hills"):  16.0,
    ("DIFC",        "DXB Airport"):  15.0,
    ("DIFC",        "Sharjah"):      27.0,
    ("Deira",       "Bur Dubai"):     4.5,
    ("Deira",       "Jumeirah"):     16.0,
    ("Deira",       "Al Quoz"):      20.0,
    ("Deira",       "Business Bay"): 12.0,
    ("Deira",       "Dubai Hills"):  28.0,
    ("Deira",       "DXB Airport"):   7.0,
    ("Deira",       "Sharjah"):      18.0,
    ("Bur Dubai",   "Jumeirah"):     13.0,
    ("Bur Dubai",   "Al Quoz"):      14.0,
    ("Bur Dubai",   "Business Bay"):  7.0,
    ("Bur Dubai",   "Dubai Hills"):  23.0,
    ("Bur Dubai",   "DXB Airport"):  12.0,
    ("Bur Dubai",   "Sharjah"):      23.0,
    ("Jumeirah",    "Al Quoz"):       8.0,
    ("Jumeirah",    "Business Bay"):  9.0,
    ("Jumeirah",    "Dubai Hills"):  16.0,
    ("Jumeirah",    "DXB Airport"):  22.0,
    ("Jumeirah",    "Sharjah"):      37.0,
    ("Al Quoz",     "Business Bay"):  9.0,
    ("Al Quoz",     "Dubai Hills"):  12.0,
    ("Al Quoz",     "DXB Airport"):  22.0,
    ("Al Quoz",     "Sharjah"):      38.0,
    ("Business Bay","Dubai Hills"):  14.0,
    ("Business Bay","DXB Airport"):  16.0,
    ("Business Bay","Sharjah"):      28.0,
    ("Dubai Hills", "DXB Airport"):  28.0,
    ("Dubai Hills", "Sharjah"):      42.0,
    ("DXB Airport", "Sharjah"):      22.0,
}

def get_dist_arr(pu_arr, do_arr):
    out = np.zeros(len(pu_arr))
    for i in range(len(pu_arr)):
        a, b = pu_arr[i], do_arr[i]
        if a == b:
            out[i] = np.random.uniform(1.5, 4.5)
        else:
            key = (a, b) if (a, b) in DIST_MATRIX else (b, a)
            base = DIST_MATRIX.get(key, 15.0)
            out[i] = base * np.random.uniform(0.90, 1.10)
    return out

# ── 3. Salik gates per zone pair ────────────────────────────────────────────
SALIK = {
    ("Marina",      "Downtown"):    2, ("Marina",      "DIFC"):        2,
    ("Marina",      "Business Bay"):2, ("Marina",      "Deira"):       3,
    ("Marina",      "Bur Dubai"):   2, ("Marina",      "DXB Airport"): 3,
    ("Marina",      "Sharjah"):     3, ("JBR",         "Downtown"):    2,
    ("JBR",         "DIFC"):        2, ("JBR",         "Business Bay"):2,
    ("JBR",         "Deira"):       3, ("JBR",         "DXB Airport"): 3,
    ("JBR",         "Sharjah"):     3, ("Al Quoz",     "Deira"):       2,
    ("Al Quoz",     "DXB Airport"): 2, ("Al Quoz",     "Sharjah"):     2,
    ("Dubai Hills", "Deira"):       2, ("Dubai Hills", "DXB Airport"): 2,
    ("Dubai Hills", "Sharjah"):     3, ("DIFC",        "DXB Airport"): 1,
    ("DIFC",        "Sharjah"):     2, ("Downtown",    "DXB Airport"): 1,
    ("Downtown",    "Sharjah"):     2, ("Deira",       "Marina"):      3,
    ("Deira",       "JBR"):         3, ("Bur Dubai",   "Marina"):      2,
    ("Bur Dubai",   "JBR"):         2, ("DXB Airport", "Marina"):      3,
    ("DXB Airport", "JBR"):         3, ("DXB Airport", "Dubai Hills"): 2,
    ("Sharjah",     "Marina"):      3, ("Sharjah",     "JBR"):         3,
    ("Sharjah",     "Dubai Hills"): 3,
}
def get_salik_arr(pu_arr, do_arr):
    out = np.zeros(len(pu_arr), dtype=int)
    for i in range(len(pu_arr)):
        a, b = pu_arr[i], do_arr[i]
        out[i] = SALIK.get((a, b), SALIK.get((b, a), 0))
    return out

# ── 4. Events calendar 2025 ─────────────────────────────────────────────────
EVENTS = [
    {"name": "Dubai Shopping Festival", "type": "Shopping Festival",
     "start": "2025-01-03", "end": "2025-02-01", "dmult": 1.35},
    {"name": "Dubai Food Festival",     "type": "Food Festival",
     "start": "2025-02-20", "end": "2025-03-08", "dmult": 1.20},
    {"name": "Art Dubai",               "type": "Art/Culture",
     "start": "2025-03-18", "end": "2025-03-23", "dmult": 1.25},
    {"name": "Dubai World Cup",         "type": "Sports Event",
     "start": "2025-03-29", "end": "2025-03-29", "dmult": 1.55},
    {"name": "Eid Al Fitr",             "type": "Religious Holiday",
     "start": "2025-03-30", "end": "2025-04-02", "dmult": 1.30},
    {"name": "Formula 1 Weekend",       "type": "Sports Event",
     "start": "2025-04-04", "end": "2025-04-06", "dmult": 1.50},
    {"name": "Eid Al Adha",             "type": "Religious Holiday",
     "start": "2025-06-05", "end": "2025-06-09", "dmult": 1.25},
    {"name": "GITEX Global",            "type": "Tech Conference",
     "start": "2025-10-13", "end": "2025-10-17", "dmult": 1.45},
    {"name": "Diwali",                  "type": "Cultural Event",
     "start": "2025-10-20", "end": "2025-10-21", "dmult": 1.20},
    {"name": "Dubai Airshow",           "type": "Trade Show",
     "start": "2025-11-17", "end": "2025-11-21", "dmult": 1.40},
    {"name": "UAE National Day",        "type": "National Holiday",
     "start": "2025-12-02", "end": "2025-12-03", "dmult": 1.35},
    {"name": "NYE Burj Khalifa",        "type": "New Year Event",
     "start": "2025-12-31", "end": "2025-12-31", "dmult": 2.20},
]
RAMADAN_START = pd.Timestamp("2025-03-01")
RAMADAN_END   = pd.Timestamp("2025-03-29")
UAE_HOLIDAYS = {
    "2025-01-01","2025-03-30","2025-03-31","2025-04-01","2025-04-02","2025-04-03",
    "2025-06-05","2025-06-06","2025-06-07","2025-06-08","2025-06-09",
    "2025-06-26","2025-09-04","2025-12-02","2025-12-03",
}

# ── 5. Products ─────────────────────────────────────────────────────────────
# Columns: base_day, base_night, per_km, per_min, min_fare,
#          book_peak, book_offpeak, book_night, is_hala(0/1), sh_reg, sh_apt
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
PD = np.array([list(v) for v in PROD_DEF.values()])   # shape (10, 11)
I_BDAY, I_BNGT, I_PKM, I_PMIN, I_MINF = 0, 1, 2, 3, 4
I_BKPK, I_BKOF, I_BKNT, I_HALA        = 5, 6, 7, 8
I_SHREG, I_SHAPT                        = 9, 10
SH_REG = PD[:, I_SHREG] / PD[:, I_SHREG].sum()
SH_APT = PD[:, I_SHAPT] / PD[:, I_SHAPT].sum()

# ── 6. Demand profiles ──────────────────────────────────────────────────────
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

# ── 7. Weather by month ─────────────────────────────────────────────────────
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

ZONE_PICK_W = np.array([1.8,1.6,1.3,1.5,1.2,1.1,1.0,0.7,1.4,0.8,1.0,0.9])
ZONE_PICK_W_APT = ZONE_PICK_W.copy()
ZONE_PICK_W_APT[ZONE_NAMES.index("DXB Airport")] *= 2.0
ZONE_PICK_W_APT /= ZONE_PICK_W_APT.sum()
ZONE_PICK_W     /= ZONE_PICK_W.sum()

# ── 8. Timestamps ───────────────────────────────────────────────────────────
print("Sampling timestamps...")
year_start = pd.Timestamp("2025-01-01")
month_ends = [31,59,90,120,151,181,212,243,273,304,334,365]
day_weights = np.zeros(365)
for m in range(12):
    ds = 0 if m==0 else month_ends[m-1]
    de = month_ends[m]
    day_weights[ds:de] = MONTH_VOL[m]
day_weights /= day_weights.sum()

day_of_year = np.random.choice(365, size=N_RIDES, p=day_weights)
timestamps_d = year_start + pd.to_timedelta(day_of_year, unit="D")
months = timestamps_d.month.values

# Ramadan mask
is_ram_day = (timestamps_d >= RAMADAN_START) & (timestamps_d <= RAMADAN_END)

# Vectorized hour sampling
hour_choice = np.zeros(N_RIDES, dtype=int)
ram_idx = np.where(is_ram_day)[0]
reg_idx = np.where(~is_ram_day)[0]
hour_choice[ram_idx] = np.random.choice(24, size=len(ram_idx), p=H_RAM)
hour_choice[reg_idx] = np.random.choice(24, size=len(reg_idx), p=H_REG)
minutes = np.random.randint(0, 60, N_RIDES)

timestamps = (
    timestamps_d
    + pd.to_timedelta(hour_choice.astype(int), unit="h")
    + pd.to_timedelta(minutes, unit="m")
)

# ── 9. Temporal flags ───────────────────────────────────────────────────────
dow = timestamps.dayofweek.values          # 0=Mon, 4=Fri, 5=Sat, 6=Sun
is_fri = (dow == 4); is_sat = (dow == 5)
is_weekend_uae = is_fri | is_sat

# RTA peak definition
is_peak = np.where(
    is_fri,
    hour_choice >= 16,
    ((hour_choice >= 8) & (hour_choice < 10)) | ((hour_choice >= 16) & (hour_choice < 20))
)
is_night  = (hour_choice >= 22) | (hour_choice < 6)
is_offpk  = (~is_peak) & (~is_night)

is_ramadan = is_ram_day.values
is_suhoor  = is_ramadan & ((hour_choice >= 1) & (hour_choice <= 3))
is_iftar   = is_ramadan & (hour_choice == 17)
date_strs  = timestamps.strftime("%Y-%m-%d")
is_holiday = np.array([d in UAE_HOLIDAYS for d in date_strs])

# ── 10. Events ──────────────────────────────────────────────────────────────
active_event = np.full(N_RIDES, "None", dtype=object)
event_type   = np.full(N_RIDES, "None", dtype=object)
event_dmult  = np.ones(N_RIDES)
for ev in EVENTS:
    mask = (timestamps_d >= pd.Timestamp(ev["start"])) & (timestamps_d <= pd.Timestamp(ev["end"]))
    active_event[mask] = ev["name"]
    event_type[mask]   = ev["type"]
    event_dmult[mask] *= ev["dmult"]

# ── 11. Zones ───────────────────────────────────────────────────────────────
print("Sampling zones...")
pu_idx = np.random.choice(len(ZONE_NAMES), size=N_RIDES, p=ZONE_PICK_W_APT)
# Dropoff: uniform over all other zones
do_idx = np.zeros(N_RIDES, dtype=int)
for i in range(N_RIDES):
    probs = ZONE_PICK_W.copy()
    probs[pu_idx[i]] = 0
    probs /= probs.sum()
    do_idx[i] = np.random.choice(len(ZONE_NAMES), p=probs)

pickup_zone  = np.array(ZONE_NAMES)[pu_idx]
dropoff_zone = np.array(ZONE_NAMES)[do_idx]
is_airport   = (pickup_zone == "DXB Airport") | (dropoff_zone == "DXB Airport")
is_intrazone = (pickup_zone == dropoff_zone)

pickup_lat  = Z_LAT[pu_idx] + np.random.normal(0, 0.008, N_RIDES)
pickup_lon  = Z_LON[pu_idx] + np.random.normal(0, 0.008, N_RIDES)
dropoff_lat = Z_LAT[do_idx] + np.random.normal(0, 0.008, N_RIDES)
dropoff_lon = Z_LON[do_idx] + np.random.normal(0, 0.008, N_RIDES)

print("Computing distances and Salik gates...")
route_distance_km = get_dist_arr(pickup_zone, dropoff_zone)
salik_gates       = get_salik_arr(pickup_zone, dropoff_zone)

# ── 12. Weather ─────────────────────────────────────────────────────────────
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

# ── 13. Products ────────────────────────────────────────────────────────────
print("Assigning products...")
prod_idx = np.where(
    is_airport,
    np.random.choice(len(PROD_NAMES), size=N_RIDES, p=SH_APT),
    np.random.choice(len(PROD_NAMES), size=N_RIDES, p=SH_REG)
)
product_type    = np.array(PROD_NAMES)[prod_idx]
is_hala_product = PD[prod_idx, I_HALA].astype(bool)

payment_method = np.random.choice(PAY_METHODS, size=N_RIDES, p=PAY_PROBS)
is_careem_plus = (payment_method == "Careem Plus")

# ── 14. Supply & trip metrics ────────────────────────────────────────────────
zone_demand = Z_DM[pu_idx] * event_dmult * weather_dmult
captain_avail = np.clip(
    1.0 - 0.35*(zone_demand - 1.0) + np.random.normal(0, 0.10, N_RIDES),
    0.15, 1.00
)
wait_base = np.where(is_airport, 5.5,
            np.where(is_peak, 4.5, 3.0))
wait_time = np.clip(
    wait_base * (1.5 - captain_avail) + np.random.exponential(1.5, N_RIDES),
    1.0, 25.0
)
avg_speed = np.where(is_peak,
    np.random.uniform(22, 30, N_RIDES),
    np.where(is_night,
        np.random.uniform(48, 65, N_RIDES),
        np.random.uniform(32, 48, N_RIDES)
    )
)
trip_duration_min = (route_distance_km / avg_speed) * 60

# ── 15. Pricing ─────────────────────────────────────────────────────────────
print("Computing fares...")
per_km_arr  = PD[prod_idx, I_PKM]
per_min_arr = PD[prod_idx, I_PMIN]
min_fare    = PD[prod_idx, I_MINF]
base_day    = PD[prod_idx, I_BDAY]
base_night  = PD[prod_idx, I_BNGT]
book_peak   = PD[prod_idx, I_BKPK]
book_off    = PD[prod_idx, I_BKOF]
book_night  = PD[prod_idx, I_BKNT]

flagfall    = np.where(is_night, base_night, base_day)
booking_fee = np.where(is_peak, book_peak,
              np.where(is_night, book_night, book_off))
salik_cost  = salik_gates * 4.0

# Hala (RTA taxi meter)
hala_apt_pu = is_hala_product & (pickup_zone == "DXB Airport")
hala_start  = np.where(hala_apt_pu, 25.0, flagfall + booking_fee)
hala_fare   = hala_start + per_km_arr * route_distance_km + wait_time * 0.50 + salik_cost
hala_final  = np.maximum(hala_fare, min_fare)

# Private hire (Careem dynamic pricing: base + per-km + per-min * full duration)
demand_supply_gap = np.clip(zone_demand - 1.0, 0.0, 1.5)
surge_mult  = np.clip(1.0 + demand_supply_gap * 0.55, 1.00, 2.50)
ph_base     = flagfall + booking_fee
ph_fare     = ph_base + (per_km_arr * route_distance_km + per_min_arr * trip_duration_min) * surge_mult + salik_cost
ph_final    = np.maximum(ph_fare, min_fare)

final_price    = np.where(is_hala_product, hala_final, ph_final)
metered_fare   = np.where(is_hala_product, hala_fare,  ph_fare)
surge_out      = np.where(is_hala_product, 1.0, surge_mult)

# ── 16. Outcomes ────────────────────────────────────────────────────────────
cancel_prob = np.clip(
    0.08
    + (1.0 - captain_avail) * 0.10
    + is_rain.astype(float) * 0.03
    + is_sandstorm.astype(float) * 0.03,
    0.05, 0.28
)
cancel_mask = np.random.random(N_RIDES) < cancel_prob
cancel_why  = np.random.choice(
    ["Captain Cancelled","Customer Cancelled","No Captain Available"],
    size=N_RIDES, p=[0.40, 0.40, 0.20]
)
status = np.full(N_RIDES, "Completed", dtype=object)
status[cancel_mask] = cancel_why[cancel_mask]
cancel_reason_col = np.full(N_RIDES, "N/A", dtype=object)
cancel_reason_col[cancel_mask] = cancel_why[cancel_mask]

capt_rating = np.round(np.random.normal(4.55, 0.30, N_RIDES).clip(1,5), 1)
cust_rating  = np.round(np.random.normal(4.62, 0.28, N_RIDES).clip(1,5), 1)
capt_rating[cancel_mask] = np.nan
cust_rating[cancel_mask]  = np.nan
eta_dev = np.round(np.random.normal(0, 1.5, N_RIDES), 1)

# ── 17. Assemble ────────────────────────────────────────────────────────────
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
    "is_weekend":                 is_weekend_uae,
    "is_peak_hour":               is_peak,
    "is_night":                   is_night,
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
    "route_distance_km":          np.round(route_distance_km, 2),
    "salik_gates":                salik_gates,
    "salik_cost_aed":             np.round(salik_cost, 2),
    "product_type":               product_type,
    "is_hala_product":            is_hala_product,
    "payment_method":             payment_method,
    "is_careem_plus":             is_careem_plus,
    "captain_availability_score": np.round(captain_avail, 3),
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

# ── 18. Validate ────────────────────────────────────────────────────────────
completed = df[df["booking_status"] == "Completed"]
apt_n = df["is_airport_ride"].sum()
print("\n── Validation ──────────────────────────────────────────────────────")
print(f"  Total rides:        {len(df):,}")
print(f"  Completed:          {len(completed):,} ({100*len(completed)/len(df):.1f}%)")
print(f"  Airport rides:      {apt_n:,} ({100*apt_n/len(df):.1f}%)")
print(f"  Avg price (compl.): AED {completed['final_price_aed'].mean():.2f}")
print(f"  Price range:        AED {df['final_price_aed'].min():.2f} - {df['final_price_aed'].max():.2f}")
print(f"  Avg distance:       {df['route_distance_km'].mean():.1f} km")
print(f"  Avg Salik gates:    {df['salik_gates'].mean():.2f}")
print(f"  Hala rides:         {df['is_hala_product'].sum():,} ({100*df['is_hala_product'].mean():.1f}%)")
print(f"\n  Product mix:")
for p in PROD_NAMES:
    g = df[df["product_type"]==p]
    print(f"    {p:<14} {len(g):>6,} ({100*len(g)/len(df):4.1f}%)  avg AED {g['final_price_aed'].mean():.2f}")

print(f"\n  Period check (20km DXB->Marina Hala Taxi, peak):")
# Manual calculation: AED 25 + 20*2.20 + avg 3min wait*0.50 + 3 Salik*4
print(f"    Expected ~AED {25 + 20*2.20 + 3*0.50 + 3*4:.1f} (AED 25 flagfall + km + wait + Salik)")

# ── 19. Save ────────────────────────────────────────────────────────────────
df.to_csv(OUT_PATH, index=False)
print(f"\n✓ Dataset saved -> {OUT_PATH}")
print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
