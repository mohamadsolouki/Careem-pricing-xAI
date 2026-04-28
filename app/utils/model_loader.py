from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dubai_rides_2025.csv"
SHAP_SAMPLE_RAW_CSV_PATH = MODELS_DIR / "shap_sample_raw.csv"
SHAP_SAMPLE_FEATURES_CSV_PATH = MODELS_DIR / "shap_sample_features.csv"
SHAP_SAMPLE_RAW_PICKLE_PATH = MODELS_DIR / "shap_sample_raw.pkl"
SHAP_SAMPLE_FEATURES_PICKLE_PATH = MODELS_DIR / "shap_sample_features.pkl"


def _path_cache_token(path: Path) -> tuple[int, int]:
    if not path.exists():
        return (0, 0)
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size)


def _paths_cache_token(*paths: Path) -> tuple[tuple[int, int], ...]:
    return tuple(_path_cache_token(path) for path in paths)


def _load_shap_frame(csv_path: Path, pickle_path: Path) -> pd.DataFrame:
    # CSV keeps the SHAP sample frames portable across different pandas builds.
    if csv_path.exists():
        return pd.read_csv(csv_path)
    with open(pickle_path, "rb") as frame_file:
        return pickle.load(frame_file)


@st.cache_resource(show_spinner=False)
def _load_model_cached(_token: tuple[int, int]):
    with open(MODELS_DIR / "xgboost_price_model.pkl", "rb") as model_file:
        return pickle.load(model_file)


def load_model():
    return _load_model_cached(_path_cache_token(MODELS_DIR / "xgboost_price_model.pkl"))


@st.cache_data(show_spinner=False)
def _load_feature_columns_cached(_token: tuple[int, int]):
    with open(MODELS_DIR / "feature_columns.pkl", "rb") as feature_file:
        return pickle.load(feature_file)


def load_feature_columns():
    return _load_feature_columns_cached(_path_cache_token(MODELS_DIR / "feature_columns.pkl"))


@st.cache_data(show_spinner=False)
def _load_metrics_cached(_token: tuple[int, int]):
    with open(MODELS_DIR / "model_metrics.json", "r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def load_metrics():
    return _load_metrics_cached(_path_cache_token(MODELS_DIR / "model_metrics.json"))


def get_interval_basis_percent(metrics: dict | None = None) -> int:
    metrics = metrics or load_metrics()
    return int(metrics.get("prediction_interval_basis_percent", 80))


def get_global_interval_half_width(metrics: dict | None = None) -> float:
    metrics = metrics or load_metrics()
    basis_percent = get_interval_basis_percent(metrics)
    value = metrics.get(f"prediction_interval_{basis_percent}_half_width")
    if value is None:
        value = metrics.get("prediction_interval_80_half_width", 0.0)
    return float(value)


def estimate_trip_interval(record: dict[str, object], metrics: dict | None = None) -> dict[str, object]:
    metrics = metrics or load_metrics()
    basis_percent = get_interval_basis_percent(metrics)
    global_half_width = get_global_interval_half_width(metrics)
    adaptive_profile = metrics.get(f"prediction_interval_adaptive_{basis_percent}")
    if not adaptive_profile:
        return {
            "basis_percent": basis_percent,
            "half_width": global_half_width,
            "label": "global-baseline",
            "is_adaptive": False,
            "score": None,
            "global_half_width": global_half_width,
        }

    weights = adaptive_profile.get("score_weights", {})
    scales = adaptive_profile.get("score_scales", {})
    event_value = str(record.get("active_event", "None") or "None")
    score = (
        float(weights.get("distance", 0.0))
        * min(max(float(record.get("route_distance_km", 0.0)) / max(float(scales.get("distance_p90", 1.0)), 1.0), 0.0), 1.5)
        + float(weights.get("traffic", 0.0))
        * min(max(max(float(record.get("traffic_index", 1.0)) - 1.0, 0.0) / max(float(scales.get("traffic_excess_p90", 0.1)), 0.1), 0.0), 1.5)
        + float(weights.get("demand", 0.0))
        * min(max(max(float(record.get("demand_index", 1.0)) - 1.0, 0.0) / max(float(scales.get("demand_excess_p90", 0.1)), 0.1), 0.0), 1.5)
        + float(weights.get("airport", 0.0)) * float(bool(record.get("is_airport_ride", False)))
        + float(weights.get("event", 0.0)) * float(event_value != "None")
        + float(weights.get("peak", 0.0)) * float(bool(record.get("is_peak_hour", False)))
    )

    bin_edges = adaptive_profile.get("bin_edges", [])
    bin_labels = adaptive_profile.get("bin_labels", [])
    bin_half_widths = adaptive_profile.get("bin_half_widths", [])
    if len(bin_edges) < 2 or not bin_labels or not bin_half_widths:
        return {
            "basis_percent": basis_percent,
            "half_width": global_half_width,
            "label": "global-baseline",
            "is_adaptive": False,
            "score": round(float(score), 3),
            "global_half_width": global_half_width,
        }

    bin_index = 0
    for edge in bin_edges[1:-1]:
        if score > float(edge):
            bin_index += 1
    bin_index = min(bin_index, len(bin_half_widths) - 1, len(bin_labels) - 1)
    return {
        "basis_percent": basis_percent,
        "half_width": float(bin_half_widths[bin_index]),
        "label": str(bin_labels[bin_index]),
        "is_adaptive": True,
        "score": round(float(score), 3),
        "global_half_width": global_half_width,
    }


@st.cache_data(show_spinner=False)
def _load_shap_bundle_cached(_token: tuple[tuple[int, int], ...]):
    with open(MODELS_DIR / "shap_values.pkl", "rb") as shap_file:
        shap_values = pickle.load(shap_file)
    sample_raw = _load_shap_frame(SHAP_SAMPLE_RAW_CSV_PATH, SHAP_SAMPLE_RAW_PICKLE_PATH)
    sample_features = _load_shap_frame(SHAP_SAMPLE_FEATURES_CSV_PATH, SHAP_SAMPLE_FEATURES_PICKLE_PATH)
    with open(MODELS_DIR / "shap_summary.json", "r", encoding="utf-8") as summary_file:
        shap_summary = json.load(summary_file)
    return {
        "values": shap_values["values"],
        "base_value": shap_values["base_value"],
        "base_values": shap_values.get("base_values"),
        "feature_names": shap_values["feature_names"],
        "sample_raw": sample_raw,
        "sample_features": sample_features,
        "summary": shap_summary,
    }


def load_shap_bundle():
    return _load_shap_bundle_cached(
        _paths_cache_token(
            MODELS_DIR / "shap_values.pkl",
            MODELS_DIR / "shap_summary.json",
            SHAP_SAMPLE_RAW_CSV_PATH,
            SHAP_SAMPLE_RAW_PICKLE_PATH,
            SHAP_SAMPLE_FEATURES_CSV_PATH,
            SHAP_SAMPLE_FEATURES_PICKLE_PATH,
        )
    )


@st.cache_data(show_spinner=False)
def _load_dataset_profile_cached(_token: tuple[int, int]):
    usecols = [
        "final_price_aed",
        "booking_status",
        "is_airport_ride",
        "pickup_zone",
        "product_type",
        "demand_index",
        "traffic_index",
    ]
    dataset = pd.read_csv(DATA_PATH, usecols=usecols)
    completed = dataset[dataset["booking_status"] == "Completed"]
    return {
        "rows": int(len(dataset)),
        "completed_rate": float((dataset["booking_status"] == "Completed").mean()),
        "avg_price": float(completed["final_price_aed"].mean()),
        "airport_share": float(dataset["is_airport_ride"].mean()),
        "avg_demand_index": float(dataset["demand_index"].mean()),
        "avg_traffic_index": float(dataset["traffic_index"].mean()),
        "top_pickup_zones": completed.groupby("pickup_zone")["final_price_aed"].mean().sort_values(ascending=False).head(5),
        "top_products": completed.groupby("product_type")["final_price_aed"].mean().sort_values(ascending=False).head(5),
    }


def load_dataset_profile():
    return _load_dataset_profile_cached(_path_cache_token(DATA_PATH))


@st.cache_data(show_spinner=False)
def _load_model_version_cached(_token: tuple[int, int]) -> dict:
    """Load model versioning metadata written by train_model.py."""
    version_path = MODELS_DIR / "model_version.json"
    if not version_path.exists():
        return {}
    with open(version_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_version() -> dict:
    return _load_model_version_cached(_path_cache_token(MODELS_DIR / "model_version.json"))