from __future__ import annotations

import os
from datetime import datetime

import requests

from utils.domain import WEATHER_PROFILES, ZONES, stable_rng


def _classify_weather(is_rain: bool, is_sandstorm: bool, humidity_pct: float) -> str:
    if is_sandstorm:
        return "Sandstorm"
    if is_rain:
        return "Rain"
    if humidity_pct >= 72:
        return "Humid"
    return "Clear"


def _mock_weather(zone_name: str, ride_dt: datetime):
    profile = WEATHER_PROFILES[ride_dt.month]
    rng = stable_rng("weather", zone_name, ride_dt.date().isoformat())
    is_rain = bool(rng.random() < profile["rain_p"])
    is_sandstorm = bool((not is_rain) and rng.random() < profile["storm_p"])
    temperature_c = float(rng.uniform(*profile["temp"]))
    humidity_pct = float(rng.uniform(*profile["hum"]))
    weather_demand_factor = 1.0 + 0.40 * float(is_rain) + 0.25 * float(is_sandstorm)
    return {
        "temperature_c": round(temperature_c, 1),
        "humidity_pct": round(humidity_pct, 1),
        "is_rain": is_rain,
        "is_sandstorm": is_sandstorm,
        "weather_demand_factor": round(weather_demand_factor, 3),
        "weather_label": _classify_weather(is_rain, is_sandstorm, humidity_pct),
        "source": "Seasonal model",
    }


def _live_weather(zone_name: str):
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return None

    zone = ZONES[zone_name]
    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            "lat": zone["lat"],
            "lon": zone["lon"],
            "appid": api_key,
            "units": "metric",
        },
        timeout=8,
    )
    response.raise_for_status()
    payload = response.json()
    weather_main = (payload.get("weather") or [{}])[0].get("main", "Clear")
    description = (payload.get("weather") or [{}])[0].get("description", "clear sky")
    temperature_c = float(payload.get("main", {}).get("temp", 30.0))
    humidity_pct = float(payload.get("main", {}).get("humidity", 60.0))
    is_rain = weather_main.lower() in {"rain", "drizzle", "thunderstorm"}
    is_sandstorm = weather_main.lower() in {"dust", "sand", "ash", "squall"} or "sand" in description.lower() or "dust" in description.lower()
    weather_demand_factor = 1.0 + 0.40 * float(is_rain) + 0.25 * float(is_sandstorm)
    return {
        "temperature_c": round(temperature_c, 1),
        "humidity_pct": round(humidity_pct, 1),
        "is_rain": is_rain,
        "is_sandstorm": is_sandstorm,
        "weather_demand_factor": round(weather_demand_factor, 3),
        "weather_label": description.title(),
        "source": "OpenWeatherMap",
    }


def get_weather(zone_name: str, ride_dt: datetime, prefer_live: bool = True):
    if prefer_live and os.getenv("OPENWEATHER_API_KEY") and ride_dt.date() == datetime.now().date():
        try:
            return _live_weather(zone_name)
        except requests.RequestException:
            pass
    return _mock_weather(zone_name, ride_dt)