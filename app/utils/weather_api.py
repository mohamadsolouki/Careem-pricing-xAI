from __future__ import annotations

from datetime import datetime

import requests

from utils.config import get_config_value
from utils.domain import WEATHER_PROFILES, get_nearest_zone, stable_rng


def _classify_weather(is_rain: bool, is_sandstorm: bool, humidity_pct: float) -> str:
    if is_sandstorm:
        return "Sandstorm"
    if is_rain:
        return "Rain"
    if humidity_pct >= 72:
        return "Humid"
    return "Clear"


def _mock_weather(lat: float, lon: float, ride_dt: datetime):
    profile = WEATHER_PROFILES[ride_dt.month]
    rng = stable_rng("weather", round(lat, 4), round(lon, 4), ride_dt.date().isoformat())
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


def _live_weather(lat: float, lon: float):
    api_key = get_config_value("OPENWEATHER_API_KEY")
    if not api_key:
        return None

    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            "lat": lat,
            "lon": lon,
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


def get_weather(lat: float, lon: float, ride_dt: datetime, prefer_live: bool = True):
    if prefer_live and get_config_value("OPENWEATHER_API_KEY") and ride_dt.date() == datetime.now().date():
        try:
            live_weather = _live_weather(lat, lon)
            live_weather["nearest_zone"] = get_nearest_zone(lat, lon)
            return live_weather
        except requests.RequestException:
            pass
    mock_weather = _mock_weather(lat, lon, ride_dt)
    mock_weather["nearest_zone"] = get_nearest_zone(lat, lon)
    return mock_weather