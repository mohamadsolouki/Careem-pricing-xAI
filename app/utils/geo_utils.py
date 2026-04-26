from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from utils.domain import ZONE_NAMES, ZONES, get_distance_km, get_salik_gates


def get_nearest_zone(lat: float, lon: float) -> str:
    zone_name = min(
        ZONE_NAMES,
        key=lambda candidate: (ZONES[candidate]["lat"] - lat) ** 2 + (ZONES[candidate]["lon"] - lon) ** 2,
    )
    return zone_name


def build_route_figure(pickup_zone: str, dropoff_zone: str) -> go.Figure:
    pickup = ZONES[pickup_zone]
    dropoff = ZONES[dropoff_zone]
    midpoint_lat = (pickup["lat"] + dropoff["lat"]) / 2
    midpoint_lon = (pickup["lon"] + dropoff["lon"]) / 2

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[pickup["lon"], dropoff["lon"]],
            y=[pickup["lat"], dropoff["lat"]],
            mode="lines",
            line={"color": "#f29f05", "width": 4},
            hoverinfo="skip",
            name="Route",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[pickup["lon"], dropoff["lon"]],
            y=[pickup["lat"], dropoff["lat"]],
            mode="markers+text",
            text=[pickup_zone, dropoff_zone],
            textposition=["top center", "bottom center"],
            marker={"size": [16, 16], "color": ["#0d9488", "#c2410c"], "line": {"color": "white", "width": 2}},
            name="Zones",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[midpoint_lon],
            y=[midpoint_lat],
            mode="markers",
            marker={"size": 28, "color": "rgba(242,159,5,0.18)", "line": {"color": "#f29f05", "width": 1}},
            hovertemplate=f"Distance: {get_distance_km(pickup_zone, dropoff_zone):.1f} km<br>Salik gates: {get_salik_gates(pickup_zone, dropoff_zone)}<extra></extra>",
            name="Route summary",
        )
    )

    lon_values = [pickup["lon"], dropoff["lon"]]
    lat_values = [pickup["lat"], dropoff["lat"]]
    figure.update_layout(
        height=360,
        margin={"l": 0, "r": 0, "t": 10, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    figure.update_xaxes(
        visible=False,
        range=[min(lon_values) - 0.04, max(lon_values) + 0.04],
    )
    figure.update_yaxes(
        visible=False,
        range=[min(lat_values) - 0.03, max(lat_values) + 0.03],
        scaleanchor="x",
        scaleratio=1,
    )
    return figure