import numpy as np
import pandas as pd


CAT_COLS = ["pickup_zone", "dropoff_zone", "product_type", "payment_method", "event_type"]

BOOL_COLS = [
    "is_weekend",
    "is_peak_hour",
    "is_late_night",
    "is_offpeak",
    "is_ramadan",
    "is_uae_public_holiday",
    "is_suhoor_window",
    "is_iftar_window",
    "is_rain",
    "is_sandstorm",
    "is_airport_ride",
    "is_intrazone_trip",
    "is_hala_product",
    "is_careem_plus",
]

TARGET = "final_price_aed"

EXCLUDE = [
    "ride_id",
    "customer_id",
    "captain_id",
    "timestamp",
    "date",
    "day_name",
    "month_name",
    "week_of_year",
    "hour",
    "day_of_week",
    "month",
    "minute",
    TARGET,
    "metered_fare_aed",
    "price_per_km_aed",
    "surge_multiplier",
    "booking_status",
    "cancellation_reason",
    "captain_rating",
    "customer_rating",
    "eta_deviation_min",
    "active_event",
    "pickup_area_type",
    "dropoff_area_type",
    "quarter",
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()

    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7)
    frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12)

    frame["rain_x_peak"] = frame["is_rain"].astype(int) * frame["is_peak_hour"].astype(int)
    frame["storm_x_peak"] = frame["is_sandstorm"].astype(int) * frame["is_peak_hour"].astype(int)
    frame["ramadan_x_hour"] = frame["is_ramadan"].astype(int) * frame["hour"]
    frame["event_x_peak"] = (frame["active_event"] != "None").astype(int) * frame["is_peak_hour"].astype(int)
    frame["airport_x_peak"] = frame["is_airport_ride"].astype(int) * frame["is_peak_hour"].astype(int)

    return frame


def encode_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = add_derived_features(df)
    categorical_cols = [column for column in CAT_COLS if column in frame.columns]
    frame = pd.get_dummies(frame, columns=categorical_cols, drop_first=False, dtype=int)

    for column in BOOL_COLS:
        if column in frame.columns:
            frame[column] = frame[column].astype(int)

    return frame


def get_feature_columns(df_encoded: pd.DataFrame) -> list[str]:
    excluded = {column for column in EXCLUDE if column in df_encoded.columns}
    return [column for column in df_encoded.columns if column not in excluded]


def prepare_training_frame(df: pd.DataFrame):
    df_encoded = encode_model_frame(df)
    feature_columns = get_feature_columns(df_encoded)
    X = df_encoded[feature_columns].fillna(0)
    y = df_encoded[TARGET]
    return X, y, feature_columns, df_encoded


def prepare_inference_frame(df: pd.DataFrame, feature_columns: list[str] | None = None):
    df_encoded = encode_model_frame(df)
    inferred_columns = get_feature_columns(df_encoded)
    X = df_encoded[inferred_columns].fillna(0)

    if feature_columns is None:
        return X, inferred_columns, df_encoded

    X = X.reindex(columns=feature_columns, fill_value=0)
    return X, feature_columns, df_encoded