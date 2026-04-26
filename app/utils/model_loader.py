from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dubai_rides_2025.csv"


@st.cache_resource(show_spinner=False)
def load_model():
    with open(MODELS_DIR / "xgboost_price_model.pkl", "rb") as model_file:
        return pickle.load(model_file)


@st.cache_data(show_spinner=False)
def load_feature_columns():
    with open(MODELS_DIR / "feature_columns.pkl", "rb") as feature_file:
        return pickle.load(feature_file)


@st.cache_data(show_spinner=False)
def load_metrics():
    with open(MODELS_DIR / "model_metrics.json", "r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


@st.cache_data(show_spinner=False)
def load_shap_bundle():
    with open(MODELS_DIR / "shap_values.pkl", "rb") as shap_file:
        shap_values = pickle.load(shap_file)
    with open(MODELS_DIR / "shap_sample_raw.pkl", "rb") as sample_raw_file:
        sample_raw = pickle.load(sample_raw_file)
    with open(MODELS_DIR / "shap_sample_features.pkl", "rb") as sample_feature_file:
        sample_features = pickle.load(sample_feature_file)
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


@st.cache_data(show_spinner=False)
def load_dataset_profile():
    usecols = [
        "final_price_aed",
        "booking_status",
        "is_airport_ride",
        "pickup_zone",
        "product_type",
        "demand_index",
        "supply_pressure_index",
    ]
    dataset = pd.read_csv(DATA_PATH, usecols=usecols)
    completed = dataset[dataset["booking_status"] == "Completed"]
    return {
        "rows": int(len(dataset)),
        "completed_rate": float((dataset["booking_status"] == "Completed").mean()),
        "avg_price": float(completed["final_price_aed"].mean()),
        "airport_share": float(dataset["is_airport_ride"].mean()),
        "avg_demand_index": float(dataset["demand_index"].mean()),
        "avg_supply_pressure": float(dataset["supply_pressure_index"].mean()),
        "top_pickup_zones": completed.groupby("pickup_zone")["final_price_aed"].mean().sort_values(ascending=False).head(5),
        "top_products": completed.groupby("product_type")["final_price_aed"].mean().sort_values(ascending=False).head(5),
    }