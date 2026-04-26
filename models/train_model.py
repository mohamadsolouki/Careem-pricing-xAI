"""
XPrice — Phase 3: ML Price Prediction Model
============================================
Trains an XGBoost regressor on the Dubai ride-hailing mirror dataset.
Saves the trained model and feature column list for use in the Streamlit app.

Usage:
    python models/train_model.py

Outputs:
    models/saved/xgboost_price_model.pkl
    models/saved/feature_columns.pkl
    models/saved/model_metrics.json
    docs/figures/feature_importance.png
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "dubai_rides_2025.csv")
SAVE_DIR  = os.path.join(BASE_DIR, "models", "saved")
FIG_DIR   = os.path.join(BASE_DIR, "docs", "figures")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

# ─── 1. Load data ─────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"  Rows: {len(df):,} | Columns: {df.shape[1]}")

# Keep only completed rides for price modelling
df = df[df["booking_status"] == "Completed"].copy()
print(f"  Completed rides for modelling: {len(df):,}")

# ─── 2. Feature Engineering ───────────────────────────────────────────────────
print("Engineering features...")

# 2a. Cyclical encoding of time (avoids ordinal bias at midnight/year wrap)
df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# 2b. Interaction features (improve SHAP interpretability)
df["rain_x_peak"]    = df["is_rain"].astype(int)    * df["is_peak_hour"].astype(int)
df["storm_x_peak"]   = df["is_sandstorm"].astype(int) * df["is_peak_hour"].astype(int)
df["ramadan_x_hour"] = df["is_ramadan"].astype(int) * df["hour"]
df["event_x_peak"]   = (df["active_event"] != "None").astype(int) * df["is_peak_hour"].astype(int)
df["airport_x_peak"] = df["is_airport_ride"].astype(int) * df["is_peak_hour"].astype(int)

# 2c. One-hot encode categoricals
CAT_COLS = ["pickup_zone", "dropoff_zone", "product_type", "payment_method", "event_type"]
df_enc = pd.get_dummies(df, columns=CAT_COLS, drop_first=False, dtype=int)

# 2d. Boolean flags → int
BOOL_COLS = [
    "is_weekend", "is_peak_hour", "is_late_night", "is_ramadan",
    "is_uae_public_holiday", "is_suhoor_window", "is_iftar_window",
    "is_rain", "is_sandstorm", "is_airport_ride", "is_intrazone_trip",
    "is_careem_plus"
]
for c in BOOL_COLS:
    if c in df_enc.columns:
        df_enc[c] = df_enc[c].astype(int)

# ─── 3. Feature Selection ─────────────────────────────────────────────────────
# Exclude identifiers, raw timestamps, outcome cols that leak the target
EXCLUDE = [
    "ride_id", "customer_id", "captain_id",
    "timestamp", "date", "day_name", "month_name",
    "week_of_year",                         # captured via month_sin/cos
    "hour", "day_of_week", "month",         # replaced by cyclical encoding
    "minute",                               # noise
    "final_price_aed",                      # TARGET
    "metered_fare_aed",                     # component of target — data leakage
    "price_per_km_aed",                     # derived from target
    "surge_multiplier",                     # direct component — leakage
    "booking_status", "cancellation_reason",# outcome cols
    "captain_rating", "customer_rating",    # post-ride outcomes
    "eta_deviation_min",                    # post-ride outcome
    "active_event",                         # replaced by event_type dummies
    "pickup_area_type", "dropoff_area_type",# captured via zone dummies
    "product_segment",                      # captured via product_type dummies
    "quarter",                              # captured via month_sin/cos
]
EXCLUDE_PRESENT = [c for c in EXCLUDE if c in df_enc.columns]

FEATURE_COLS = [c for c in df_enc.columns if c not in EXCLUDE_PRESENT]
TARGET = "final_price_aed"

X = df_enc[FEATURE_COLS].fillna(0)
y = df_enc[TARGET]

print(f"  Features: {len(FEATURE_COLS)}")
print(f"  Target range: AED {y.min():.2f} – {y.max():.2f} | Mean: {y.mean():.2f}")

# ─── 4. Train / Test Split (stratified by month) ─────────────────────────────
print("Splitting train/test by month...")
# Assign month back from df_enc (already dropped "month" col in EXCLUDE,
# but we kept the cyclical versions; use original df for stratification label)
month_labels = df["month"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=month_labels
)
print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

# ─── 5. XGBoost Training ─────────────────────────────────────────────────────
print("Training XGBoost regressor...")

params = {
    "n_estimators":       800,
    "max_depth":          7,
    "learning_rate":      0.05,
    "subsample":          0.85,
    "colsample_bytree":   0.80,
    "min_child_weight":   5,
    "reg_alpha":          0.5,
    "reg_lambda":         1.5,
    "objective":          "reg:squarederror",
    "tree_method":        "hist",          # fast CPU training
    "random_state":       42,
    "n_jobs":             -1,
    "early_stopping_rounds": 30,
    "eval_metric":        "rmse",
    "verbosity":          0,
}

model = xgb.XGBRegressor(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

# ─── 6. Evaluation ────────────────────────────────────────────────────────────
print("Evaluating model...")
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

def calc_metrics(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"  [{label}] RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

train_metrics = calc_metrics(y_train, y_pred_train, "TRAIN")
test_metrics  = calc_metrics(y_test,  y_pred_test,  "TEST ")

metrics = {
    "train": {k: round(float(v), 4) for k, v in train_metrics.items()},
    "test":  {k: round(float(v), 4) for k, v in test_metrics.items()},
    "n_features": len(FEATURE_COLS),
    "n_train": int(len(X_train)),
    "n_test":  int(len(X_test)),
    "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else params["n_estimators"],
}

metrics_path = os.path.join(SAVE_DIR, "model_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  Metrics saved → {metrics_path}")

# ─── 7. Feature Importance Plot ───────────────────────────────────────────────
print("Plotting feature importance (top 30)...")
importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
top30 = importance.nlargest(30).sort_values()

fig, ax = plt.subplots(figsize=(9, 8))
colors = ["#1B4F8A" if v >= top30.quantile(0.7) else "#5B9BD5" for v in top30]
top30.plot(kind="barh", ax=ax, color=colors, edgecolor="white", linewidth=0.4)
ax.set_xlabel("Feature Importance (gain)", fontsize=11)
ax.set_title("XGBoost — Top 30 Feature Importances\nXPrice Dubai Ride-Hailing Pricing Model", fontsize=12, fontweight="bold")
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", labelsize=8)
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "feature_importance.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {fig_path}")

# ─── 8. Prediction vs Actual Scatter ──────────────────────────────────────────
print("Plotting prediction vs actual...")
sample_idx = np.random.choice(len(y_test), min(3000, len(y_test)), replace=False)
y_samp = np.array(y_test)[sample_idx]
p_samp = y_pred_test[sample_idx]

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_samp, p_samp, alpha=0.25, s=8, color="#1B4F8A", label="Predictions")
lims = [max(0, y.min() - 5), y.max() + 5]
ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect fit")
ax.set_xlabel("Actual Price (AED)", fontsize=11)
ax.set_ylabel("Predicted Price (AED)", fontsize=11)
ax.set_title(f"XGBoost — Predicted vs Actual Fare\nTest R² = {test_metrics['r2']:.4f}  |  RMSE = {test_metrics['rmse']:.2f} AED", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
scatter_path = os.path.join(FIG_DIR, "pred_vs_actual.png")
plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {scatter_path}")

# ─── 9. Save Model & Feature Columns ─────────────────────────────────────────
print("Saving model artefacts...")
model_path = os.path.join(SAVE_DIR, "xgboost_price_model.pkl")
feat_path  = os.path.join(SAVE_DIR, "feature_columns.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)
with open(feat_path, "wb") as f:
    pickle.dump(FEATURE_COLS, f)

print(f"  Model saved  → {model_path}")
print(f"  Features saved → {feat_path}")

print("\n✓ Phase 3 complete. Model is ready for SHAP analysis.")
print(f"  Test R² = {test_metrics['r2']:.4f}  |  Test RMSE = {test_metrics['rmse']:.2f} AED  |  Test MAE = {test_metrics['mae']:.2f} AED")
