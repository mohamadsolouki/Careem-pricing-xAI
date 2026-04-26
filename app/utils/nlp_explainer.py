from __future__ import annotations

from collections import defaultdict

import pandas as pd


def _feature_group(feature_name: str, record: dict[str, object]) -> str:
    if feature_name in {"route_distance_km", "route_direct_distance_km", "distance_gap_km", "route_efficiency_ratio"}:
        return "Distance"
    if feature_name in {"salik_gates", "salik_cost_aed"}:
        return "Toll roads"
    if feature_name in {"demand_index", "event_demand_multiplier", "event_x_peak"} or feature_name.startswith("event_type_"):
        return "Event and city demand"
    if feature_name in {"captain_availability_score", "supply_pressure_index", "wait_time_min", "airport_x_peak", "pickup_density_score", "dropoff_density_score"}:
        return "Captain availability"
    if feature_name in {"is_peak_hour", "is_late_night", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "ramadan_x_hour"}:
        return "Time of day"
    if feature_name in {"is_rain", "is_sandstorm", "weather_demand_factor", "rain_x_peak", "storm_x_peak", "temperature_c", "humidity_pct"}:
        return "Weather"
    if feature_name in {"booking_fee_aed", "is_hala_product"} or feature_name.startswith("product_type_"):
        return "Selected product"
    if feature_name in {"pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon", "route_mid_lat", "route_mid_lon", "lat_delta", "lon_delta", "route_bearing_deg", "bearing_sin", "bearing_cos", "is_airport_ride", "is_intrazone_trip"}:
        return "Route geography"
    if feature_name in {"trip_duration_min", "avg_speed_kmh", "traffic_index", "traffic_x_peak", "demand_x_traffic", "efficiency_x_traffic"}:
        return "Traffic conditions"
    return feature_name.replace("_", " ").title()


def summarise_contributions(contribution_series: pd.Series, record: dict[str, object]) -> pd.DataFrame:
    grouped: dict[str, float] = defaultdict(float)
    for feature_name, contribution in contribution_series.items():
        grouped[_feature_group(feature_name, record)] += float(contribution)
    summary = (
        pd.DataFrame(
            [{"driver": driver, "contribution_aed": round(value, 2), "abs_contribution": abs(value)} for driver, value in grouped.items()]
        )
        .sort_values("abs_contribution", ascending=False)
        .reset_index(drop=True)
    )
    return summary.drop(columns=["abs_contribution"])


def build_explanation(record: dict[str, object], contribution_series: pd.Series, predicted_price: float, base_value: float) -> dict[str, object]:
    summary = summarise_contributions(contribution_series, record)
    top_positive = summary[summary["contribution_aed"] > 0].head(3)
    top_negative = summary[summary["contribution_aed"] < 0].head(2)

    event_name = record.get("active_event", "None")
    weather_label = record.get("weather_label", "Clear")
    route_line = f"{record['pickup_zone']} to {record['dropoff_zone']}"
    headline = f"Estimated fare for {route_line}: AED {predicted_price:,.2f}."

    sentences = [
        f"The model starts from a baseline of AED {base_value:,.2f} and then adjusts for this route, product, and current demand conditions.",
    ]
    if not top_positive.empty:
        lifts = ", ".join(f"{row.driver} (+AED {row.contribution_aed:,.2f})" for row in top_positive.itertuples())
        sentences.append(f"The strongest upward drivers are {lifts}.")
    if not top_negative.empty:
        reductions = ", ".join(f"{row.driver} (AED {abs(row.contribution_aed):,.2f} lower)" for row in top_negative.itertuples())
        sentences.append(f"The main factors keeping the fare lower are {reductions}.")
    if event_name != "None":
        sentences.append(f"Event overlay: {event_name} is active, which raises demand pressure around the city.")
    sentences.append(
        f"Weather context: {weather_label} from the {record.get('weather_source', 'seasonal')} feed. Traffic is {record.get('traffic_condition', 'moderate').lower()} from the {record.get('traffic_source', 'synthetic')} source, with demand index {record['demand_index']:.2f} and availability score {record['captain_availability_score']:.2f}."
    )

    return {
        "headline": headline,
        "sentences": sentences,
        "summary_table": summary,
    }