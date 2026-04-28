import json
import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

from feature_engineering import prepare_inference_frame


warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "dubai_rides_2025.csv")
SAVE_DIR = os.path.join(BASE_DIR, "models", "saved")
FIG_DIR = os.path.join(BASE_DIR, "docs", "figures", "shap")

MODEL_PATH = os.path.join(SAVE_DIR, "xgboost_price_model.pkl")
FEATURE_PATH = os.path.join(SAVE_DIR, "feature_columns.pkl")
SHAP_VALUES_PATH = os.path.join(SAVE_DIR, "shap_values.pkl")
SHAP_SAMPLE_RAW_PATH = os.path.join(SAVE_DIR, "shap_sample_raw.pkl")
SHAP_SAMPLE_FEATURES_PATH = os.path.join(SAVE_DIR, "shap_sample_features.pkl")
SHAP_SAMPLE_RAW_CSV_PATH = os.path.join(SAVE_DIR, "shap_sample_raw.csv")
SHAP_SAMPLE_FEATURES_CSV_PATH = os.path.join(SAVE_DIR, "shap_sample_features.csv")
SHAP_SUMMARY_PATH = os.path.join(SAVE_DIR, "shap_summary.json")

os.makedirs(FIG_DIR, exist_ok=True)

SAMPLE_SIZE = 5000
TOP_FEATURES = 20


def save_current_figure(path: str, width: float = 12, height: float = 7):
    fig = plt.gcf()
    fig.set_size_inches(width, height)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def percentile_scale(series: pd.Series) -> np.ndarray:
    lower = np.nanpercentile(series, 5)
    upper = np.nanpercentile(series, 95)
    if np.isclose(lower, upper):
        return np.full(len(series), 0.5)
    clipped = np.clip(series, lower, upper)
    return (clipped - lower) / (upper - lower)


def plot_beeswarm(contrib_values: np.ndarray, feature_frame: pd.DataFrame, path: str):
    mean_abs = np.abs(contrib_values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:TOP_FEATURES]
    ordered_indices = top_indices[::-1]

    plt.figure()
    axis = plt.gca()
    for row_index, feature_index in enumerate(ordered_indices):
        feature_name = feature_frame.columns[feature_index]
        x_values = contrib_values[:, feature_index]
        y_values = np.random.normal(loc=row_index, scale=0.10, size=len(x_values))
        color_scale = percentile_scale(feature_frame.iloc[:, feature_index])
        scatter = axis.scatter(
            x_values,
            y_values,
            c=color_scale,
            cmap="coolwarm",
            s=8,
            alpha=0.45,
            linewidths=0,
        )

    axis.axvline(0, color="#555555", linestyle="--", linewidth=1)
    axis.set_yticks(range(len(ordered_indices)))
    axis.set_yticklabels([feature_frame.columns[index] for index in ordered_indices], fontsize=9)
    axis.set_xlabel("Feature contribution to price (AED)")
    axis.set_title("XPrice Contribution Beeswarm")
    axis.spines[["top", "right"]].set_visible(False)
    colorbar = plt.colorbar(scatter, ax=axis, pad=0.01)
    colorbar.set_label("Relative feature value")
    save_current_figure(path, width=12, height=8)


def plot_bar(contrib_values: np.ndarray, feature_names: list[str], path: str):
    mean_abs = np.abs(contrib_values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:TOP_FEATURES]
    ordered_indices = top_indices[::-1]

    plt.figure()
    axis = plt.gca()
    axis.barh(
        [feature_names[index] for index in ordered_indices],
        mean_abs[ordered_indices],
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.6,
    )
    axis.set_xlabel("Mean absolute contribution (AED)")
    axis.set_title("XPrice Global Feature Importance")
    axis.spines[["top", "right"]].set_visible(False)
    save_current_figure(path, width=10, height=7)


def plot_dependence(contrib_values: np.ndarray, feature_frame: pd.DataFrame, feature_name: str, path: str):
    feature_index = feature_frame.columns.get_loc(feature_name)
    x_values = feature_frame[feature_name]
    y_values = contrib_values[:, feature_index]
    color_scale = percentile_scale(x_values)

    plt.figure()
    axis = plt.gca()
    scatter = axis.scatter(x_values, y_values, c=color_scale, cmap="viridis", s=12, alpha=0.45, linewidths=0)
    axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
    axis.set_xlabel(feature_name)
    axis.set_ylabel("Feature contribution (AED)")
    axis.set_title(f"Contribution dependence: {feature_name}")
    axis.spines[["top", "right"]].set_visible(False)
    colorbar = plt.colorbar(scatter, ax=axis, pad=0.01)
    colorbar.set_label("Relative feature value")
    save_current_figure(path, width=10, height=7)


print("Loading model artefacts...")
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)
with open(FEATURE_PATH, "rb") as feature_file:
    feature_columns = pickle.load(feature_file)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df = df[df["booking_status"] == "Completed"].copy()
print(f"  Completed rides available: {len(df):,}")

sample_size = min(SAMPLE_SIZE, len(df))
_, sample_raw = train_test_split(
    df,
    test_size=sample_size,
    random_state=42,
    stratify=df["month"],
)
sample_raw = sample_raw.reset_index(drop=True)
print(f"  SHAP sample size: {len(sample_raw):,}")

print("Preparing feature matrix...")
X_sample, _, _ = prepare_inference_frame(sample_raw, feature_columns)

print("Computing tree contribution values...")
booster = model.get_booster()
dmatrix = xgb.DMatrix(X_sample, feature_names=feature_columns)
contrib_matrix = booster.predict(dmatrix, pred_contribs=True)
contrib_values = contrib_matrix[:, :-1]
base_values = contrib_matrix[:, -1]
base_value = float(np.mean(base_values))

print("Saving SHAP artefacts...")
with open(SHAP_VALUES_PATH, "wb") as shap_file:
    pickle.dump(
        {
            "values": contrib_values,
            "base_value": float(base_value),
            "base_values": base_values,
            "feature_names": feature_columns,
        },
        shap_file,
    )
with open(SHAP_SAMPLE_RAW_PATH, "wb") as sample_raw_file:
    pickle.dump(sample_raw, sample_raw_file)
with open(SHAP_SAMPLE_FEATURES_PATH, "wb") as sample_features_file:
    pickle.dump(X_sample, sample_features_file)
sample_raw.to_csv(SHAP_SAMPLE_RAW_CSV_PATH, index=False)
X_sample.to_csv(SHAP_SAMPLE_FEATURES_CSV_PATH, index=False)

mean_abs_shap = np.abs(contrib_values).mean(axis=0)
top_indices = np.argsort(mean_abs_shap)[::-1][:TOP_FEATURES]
summary = [
    {
        "feature": feature_columns[index],
        "mean_abs_shap": round(float(mean_abs_shap[index]), 4),
    }
    for index in top_indices
]
with open(SHAP_SUMMARY_PATH, "w", encoding="utf-8") as summary_file:
    json.dump(summary, summary_file, indent=2)

print("Rendering SHAP figures...")
plot_beeswarm(contrib_values, X_sample, os.path.join(FIG_DIR, "shap_beeswarm.png"))
plot_bar(contrib_values, feature_columns, os.path.join(FIG_DIR, "shap_bar.png"))

for feature_name, file_name in [
    ("route_distance_km", "shap_dependence_distance.png"),
    ("demand_index", "shap_dependence_demand.png"),
    ("traffic_index", "shap_dependence_traffic.png"),
    ("distance_x_traffic", "shap_dependence_dist_traffic.png"),
]:
    if feature_name in X_sample.columns:
        plot_dependence(contrib_values, X_sample, feature_name, os.path.join(FIG_DIR, file_name))

print("✓ SHAP analysis complete.")
print(f"  Saved → {SHAP_VALUES_PATH}")
print(f"  Saved → {SHAP_SAMPLE_RAW_PATH}")
print(f"  Saved → {SHAP_SAMPLE_FEATURES_PATH}")
print(f"  Saved → {SHAP_SAMPLE_RAW_CSV_PATH}")
print(f"  Saved → {SHAP_SAMPLE_FEATURES_CSV_PATH}")
print(f"  Saved → {SHAP_SUMMARY_PATH}")