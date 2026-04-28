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
import hashlib
import warnings
import datetime as _dt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

INTERVAL_TARGET_PERCENT = 80
ADAPTIVE_INTERVAL_BIN_LABELS = [
    "lower-complexity",
    "mid-complexity",
    "higher-complexity",
]
ADAPTIVE_INTERVAL_WEIGHTS = {
    "distance": 0.35,
    "traffic": 0.25,
    "demand": 0.20,
    "airport": 0.10,
    "event": 0.05,
    "peak": 0.05,
}


def _compute_uncertainty_score(
    frame: pd.DataFrame,
    score_scales: dict[str, float],
    score_weights: dict[str, float],
) -> np.ndarray:
    distance_scale = max(float(score_scales["distance_p90"]), 1.0)
    traffic_scale = max(float(score_scales["traffic_excess_p90"]), 0.1)
    demand_scale = max(float(score_scales["demand_excess_p90"]), 0.1)
    event_flag = frame["active_event"].fillna("None").astype(str).ne("None").astype(float)

    score = (
        score_weights["distance"]
        * np.clip(frame["route_distance_km"].astype(float) / distance_scale, 0.0, 1.5)
        + score_weights["traffic"]
        * np.clip((frame["traffic_index"].astype(float) - 1.0).clip(lower=0.0) / traffic_scale, 0.0, 1.5)
        + score_weights["demand"]
        * np.clip((frame["demand_index"].astype(float) - 1.0).clip(lower=0.0) / demand_scale, 0.0, 1.5)
        + score_weights["airport"] * frame["is_airport_ride"].astype(float)
        + score_weights["event"] * event_flag
        + score_weights["peak"] * frame["is_peak_hour"].astype(float)
    )
    return np.asarray(score, dtype=float)


def _build_adaptive_interval_profile(
    validation_frame: pd.DataFrame,
    validation_abs_residuals: np.ndarray,
    test_frame: pd.DataFrame,
    test_abs_residuals: np.ndarray,
    target_percent: int,
) -> dict[str, object]:
    score_scales = {
        "distance_p90": max(float(validation_frame["route_distance_km"].quantile(0.90)), 1.0),
        "traffic_excess_p90": max(
            float((validation_frame["traffic_index"] - 1.0).clip(lower=0.0).quantile(0.90)),
            0.1,
        ),
        "demand_excess_p90": max(
            float((validation_frame["demand_index"] - 1.0).clip(lower=0.0).quantile(0.90)),
            0.1,
        ),
    }
    validation_scores = _compute_uncertainty_score(
        validation_frame,
        score_scales=score_scales,
        score_weights=ADAPTIVE_INTERVAL_WEIGHTS,
    )
    bin_edges = np.quantile(validation_scores, [0.0, 0.33, 0.66, 1.0]).astype(float)
    for idx in range(1, len(bin_edges)):
        if bin_edges[idx] <= bin_edges[idx - 1]:
            bin_edges[idx] = bin_edges[idx - 1] + 1e-6

    validation_bins = np.digitize(validation_scores, bin_edges[1:-1], right=True)
    test_scores = _compute_uncertainty_score(
        test_frame,
        score_scales=score_scales,
        score_weights=ADAPTIVE_INTERVAL_WEIGHTS,
    )
    test_bins = np.digitize(test_scores, bin_edges[1:-1], right=True)

    global_half_width = float(np.percentile(validation_abs_residuals, target_percent))
    adaptive_half_widths = np.full(len(test_abs_residuals), global_half_width, dtype=float)
    bin_half_widths = []
    bin_validation_counts = []
    bin_test_counts = []
    bin_test_coverages = []

    for idx, _label in enumerate(ADAPTIVE_INTERVAL_BIN_LABELS):
        validation_mask = validation_bins == idx
        half_width = global_half_width
        if np.any(validation_mask):
            half_width = float(np.percentile(validation_abs_residuals[validation_mask], target_percent))
        bin_half_widths.append(round(float(half_width), 2))
        bin_validation_counts.append(int(np.sum(validation_mask)))

        test_mask = test_bins == idx
        adaptive_half_widths[test_mask] = half_width
        bin_test_counts.append(int(np.sum(test_mask)))
        if np.any(test_mask):
            bin_test_coverages.append(round(float(np.mean(test_abs_residuals[test_mask] <= half_width)), 4))
        else:
            bin_test_coverages.append(None)

    adaptive_test_coverage = float(np.mean(test_abs_residuals <= adaptive_half_widths))
    adaptive_mean_half_width_on_test = float(np.mean(adaptive_half_widths))
    return {
        "target_percent": int(target_percent),
        "bin_labels": ADAPTIVE_INTERVAL_BIN_LABELS,
        "bin_edges": [round(float(value), 4) for value in bin_edges],
        "bin_half_widths": bin_half_widths,
        "bin_validation_counts": bin_validation_counts,
        "bin_test_counts": bin_test_counts,
        "bin_test_coverages": bin_test_coverages,
        "score_weights": {key: round(float(value), 4) for key, value in ADAPTIVE_INTERVAL_WEIGHTS.items()},
        "score_scales": {key: round(float(value), 4) for key, value in score_scales.items()},
        "adaptive_test_coverage": round(adaptive_test_coverage, 4),
        "adaptive_mean_half_width_on_test": round(adaptive_mean_half_width_on_test, 2),
        "global_baseline_half_width": round(global_half_width, 2),
    }

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

# ─── 4. Chronological Train / Validation / Test Split ────────────────────────
print("Splitting train/validation/test chronologically by month...")
sorted_months = [int(month) for month in sorted(df["month"].unique())]
train_months = sorted_months[:8]
validation_months = sorted_months[8:10]
test_months = sorted_months[10:]

train_mask = np.isin(month_labels, train_months)
validation_mask = np.isin(month_labels, validation_months)
test_mask = np.isin(month_labels, test_months)

X_train = X.loc[train_mask].reset_index(drop=True)
y_train = y.loc[train_mask].reset_index(drop=True)
X_val = X.loc[validation_mask].reset_index(drop=True)
y_val = y.loc[validation_mask].reset_index(drop=True)
X_test = X.loc[test_mask].reset_index(drop=True)
y_test = y.loc[test_mask].reset_index(drop=True)
validation_frame = df.loc[validation_mask].reset_index(drop=True)
test_frame = df.loc[test_mask].reset_index(drop=True)

print(f"  Train months: {train_months} → {len(X_train):,} rows")
print(f"  Validation months: {validation_months} → {len(X_val):,} rows")
print(f"  Test months: {test_months} → {len(X_test):,} rows")

# ─── 4b. Forward month CV (5 folds, descriptive only) ───────────────────────
print("Running 5-fold forward month CV...")
cv_r2_scores = []
for _fold, _test_month in enumerate(sorted_months[-5:], start=1):
    _train_months = [month for month in sorted_months if month < _test_month]
    _tr_mask = np.isin(month_labels, _train_months)
    _te_mask = month_labels == _test_month
    if _tr_mask.sum() == 0 or _te_mask.sum() == 0:
        break
    _cv_model = xgb.XGBRegressor(
        n_estimators=400, max_depth=7, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.80, min_child_weight=5,
        reg_alpha=0.5, reg_lambda=1.5, objective="reg:squarederror",
        tree_method="hist", random_state=42, n_jobs=-1, verbosity=0,
    )
    _cv_model.fit(X.iloc[_tr_mask], y.iloc[_tr_mask])
    _cv_pred = _cv_model.predict(X.iloc[_te_mask])
    _fold_r2 = r2_score(y.iloc[_te_mask], _cv_pred)
    cv_r2_scores.append(_fold_r2)
    print(f"  Fold {_fold}: train months {_train_months[0]}–{_train_months[-1]}, "
          f"test month [{_test_month}] → R²={_fold_r2:.4f}")
if cv_r2_scores:
    print(f"  CV R² mean={np.mean(cv_r2_scores):.4f}  std={np.std(cv_r2_scores):.4f}  "
          f"min={np.min(cv_r2_scores):.4f}  max={np.max(cv_r2_scores):.4f}")

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
    eval_set=[(X_val, y_val)],
    verbose=False,
)

# ─── 6. Evaluation ────────────────────────────────────────────────────────────
print("Evaluating model...")
y_pred_train = model.predict(X_train)
y_pred_val   = model.predict(X_val)
y_pred_test  = model.predict(X_test)

def calc_metrics(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"  [{label}] RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

train_metrics = calc_metrics(y_train, y_pred_train, "TRAIN")
validation_metrics = calc_metrics(y_val, y_pred_val, "VALID")
test_metrics  = calc_metrics(y_test,  y_pred_test,  "TEST ")

# Prediction intervals: global baseline on validation, plus adaptive per-trip bins.
_val_abs_residuals = np.abs(np.array(y_val) - y_pred_val)
_test_abs_residuals = np.abs(np.array(y_test) - y_pred_test)
_pi80_half_width = float(np.percentile(_val_abs_residuals, INTERVAL_TARGET_PERCENT))
_pi80_coverage = float(np.mean(_test_abs_residuals <= _pi80_half_width))
_adaptive_interval_80 = _build_adaptive_interval_profile(
    validation_frame=validation_frame,
    validation_abs_residuals=_val_abs_residuals,
    test_frame=test_frame,
    test_abs_residuals=_test_abs_residuals,
    target_percent=INTERVAL_TARGET_PERCENT,
)
print(f"  Global {INTERVAL_TARGET_PERCENT}% PI half-width: ±AED {_pi80_half_width:.2f}  "
      f"(calibrated on validation, coverage on test: {_pi80_coverage * 100:.1f}%)")
print(
    "  Adaptive 80% trip bands: "
    f"{ADAPTIVE_INTERVAL_BIN_LABELS[0]} ±AED {_adaptive_interval_80['bin_half_widths'][0]:.2f} | "
    f"{ADAPTIVE_INTERVAL_BIN_LABELS[1]} ±AED {_adaptive_interval_80['bin_half_widths'][1]:.2f} | "
    f"{ADAPTIVE_INTERVAL_BIN_LABELS[2]} ±AED {_adaptive_interval_80['bin_half_widths'][2]:.2f} "
    f"(overall held-out coverage: {_adaptive_interval_80['adaptive_test_coverage'] * 100:.1f}%)"
)

metrics = {
    "train": {k: round(float(v), 4) for k, v in train_metrics.items()},
    "validation": {k: round(float(v), 4) for k, v in validation_metrics.items()},
    "test":  {k: round(float(v), 4) for k, v in test_metrics.items()},
    "cv": {
        "r2_scores": [round(float(v), 4) for v in cv_r2_scores],
        "r2_mean":   round(float(np.mean(cv_r2_scores)), 4) if cv_r2_scores else None,
        "r2_std":    round(float(np.std(cv_r2_scores)),  4) if cv_r2_scores else None,
    },
    "split": {
        "strategy": "chronological-month-train-validation-test",
        "train_months": train_months,
        "validation_months": validation_months,
        "test_months": test_months,
    },
    "n_features": len(FEATURE_COLS),
    "n_train": int(len(X_train)),
    "n_validation": int(len(X_val)),
    "n_test":  int(len(X_test)),
    "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else params["n_estimators"],
    "prediction_interval_basis_percent": INTERVAL_TARGET_PERCENT,
    "prediction_interval_80_half_width": round(_pi80_half_width, 2),
    "prediction_interval_80_coverage":   round(_pi80_coverage, 4),
    "prediction_interval_adaptive_80": _adaptive_interval_80,
}

metrics_path = os.path.join(SAVE_DIR, "model_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  Metrics saved → {metrics_path}")
# ─── Model versioning metadata ────────────────────────────────────────────────
_dataset_hash = hashlib.md5(open(DATA_PATH, "rb").read(1 << 20)).hexdigest()  # first 1 MB for speed
try:
    import subprocess
    _git_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, stderr=subprocess.DEVNULL,
    ).decode().strip()
    _git_branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=BASE_DIR, stderr=subprocess.DEVNULL,
    ).decode().strip()
    _git_status_lines = subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=no"], cwd=BASE_DIR, stderr=subprocess.DEVNULL,
    ).decode().splitlines()
    _git_worktree_dirty = bool(_git_status_lines)
    _git_dirty_paths = [line[3:] for line in _git_status_lines if len(line) > 3]
except Exception:
    _git_commit = "unknown"
    _git_branch = "unknown"
    _git_worktree_dirty = None
    _git_dirty_paths = []

_version_meta = {
    "training_date":  _dt.datetime.now().isoformat(timespec="seconds"),
    "dataset_path":   DATA_PATH,
    "dataset_hash_md5_first1MB": _dataset_hash,
    "git_commit":     _git_commit,
    "git_branch":     _git_branch,
    "git_worktree_dirty": _git_worktree_dirty,
    "git_dirty_paths": _git_dirty_paths,
    "n_rows_total":   len(df),
    "n_train":        int(len(X_train)),
    "n_validation":   int(len(X_val)),
    "n_test":         int(len(X_test)),
    "split_strategy": "chronological-month-train-validation-test",
    "n_features":     len(FEATURE_COLS),
    "validation_r2":  metrics["validation"]["r2"],
    "test_r2":        metrics["test"]["r2"],
    "cv_r2_mean":     metrics["cv"]["r2_mean"],
    "interval_basis_percent": INTERVAL_TARGET_PERCENT,
    "prediction_interval_80_half_width": metrics["prediction_interval_80_half_width"],
    "prediction_interval_80_coverage": metrics["prediction_interval_80_coverage"],
    "prediction_interval_adaptive_80_coverage": metrics["prediction_interval_adaptive_80"]["adaptive_test_coverage"],
}
version_path = os.path.join(SAVE_DIR, "model_version.json")
with open(version_path, "w") as f:
    json.dump(_version_meta, f, indent=2)
print(f"  Version metadata saved \u2192 {version_path}")
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
