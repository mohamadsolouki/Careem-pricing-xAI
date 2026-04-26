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
import xgboost as xgb

from feature_engineering import TARGET, prepare_training_frame

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

month_labels = df["month"].values

# ─── 2. Feature Engineering ───────────────────────────────────────────────────
print("Engineering features...")
X, y, FEATURE_COLS, df_enc = prepare_training_frame(df)

print(f"  Features: {len(FEATURE_COLS)}")
print(f"  Target range: AED {y.min():.2f} – {y.max():.2f} | Mean: {y.mean():.2f}")

# ─── 4. Train / Test Split (stratified by month) ─────────────────────────────
print("Splitting train/test by month...")
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
