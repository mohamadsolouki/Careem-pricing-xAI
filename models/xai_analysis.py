import json
import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

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

print("Computing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
base_value = explainer.expected_value
if isinstance(base_value, np.ndarray):
    base_value = float(np.ravel(base_value)[0])

print("Saving SHAP artefacts...")
with open(SHAP_VALUES_PATH, "wb") as shap_file:
    pickle.dump(
        {
            "values": shap_values,
            "base_value": float(base_value),
            "feature_names": feature_columns,
        },
        shap_file,
    )
with open(SHAP_SAMPLE_RAW_PATH, "wb") as sample_raw_file:
    pickle.dump(sample_raw, sample_raw_file)
with open(SHAP_SAMPLE_FEATURES_PATH, "wb") as sample_features_file:
    pickle.dump(X_sample, sample_features_file)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
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
shap.summary_plot(shap_values, X_sample, show=False, max_display=TOP_FEATURES)
save_current_figure(os.path.join(FIG_DIR, "shap_beeswarm.png"), width=12, height=8)

shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=TOP_FEATURES)
save_current_figure(os.path.join(FIG_DIR, "shap_bar.png"), width=10, height=7)

for feature_name, file_name in [
    ("route_distance_km", "shap_dependence_distance.png"),
    ("demand_index", "shap_dependence_demand.png"),
    ("captain_availability_score", "shap_dependence_supply.png"),
]:
    if feature_name in X_sample.columns:
        shap.dependence_plot(feature_name, shap_values, X_sample, interaction_index=None, show=False)
        save_current_figure(os.path.join(FIG_DIR, file_name), width=10, height=7)

print("✓ SHAP analysis complete.")
print(f"  Saved → {SHAP_VALUES_PATH}")
print(f"  Saved → {SHAP_SAMPLE_RAW_PATH}")
print(f"  Saved → {SHAP_SAMPLE_FEATURES_PATH}")
print(f"  Saved → {SHAP_SUMMARY_PATH}")