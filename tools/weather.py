# tools/weather.py
from __future__ import annotations
from typing import Any, Dict
import requests

from tools.registry import register_tool

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODE_ZH = {
    0: "晴", 1: "大部晴朗", 2: "局部多云", 3: "阴",
    45: "雾", 48: "雾凇",
    51: "小毛毛雨", 53: "中等毛毛雨", 55: "较强毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "阵雨（弱）", 81: "阵雨（中）", 82: "阵雨（强）",
    95: "雷暴",
}

def _safe_get(d: Dict[str, Any], key: str, default=None):
    return d.get(key, default) if isinstance(d, dict) else default


@register_tool(
    name="get_weather",
    description="查询指定城市的实时天气（Open-Meteo，无需API Key）。返回包含天气现象、气温、体感、降水、风速、观测时间等字段。",
    schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市中文名或英文名，例如：北京 / Shanghai / Shenzhen"}
        },
        "required": ["city"]
    }
)
def get_weather(city: str) -> Dict[str, Any]:
    city = (city or "").strip()
    if not city:
        raise ValueError("city 不能为空")

    geo_params = {"name": city, "count": 1, "language": "zh", "format": "json"}
    geo_resp = requests.get(GEOCODING_URL, params=geo_params, timeout=15)
    geo_resp.raise_for_status()
    geo_data = geo_resp.json()
    results = geo_data.get("results") or []
    if not results:
        raise ValueError(f"Geocoding 未找到城市：{city}")

    top = results[0]
    lat = float(top["latitude"])
    lon = float(top["longitude"])

    forecast_params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
        "timezone": "auto",
    }
    fc_resp = requests.get(FORECAST_URL, params=forecast_params, timeout=15)
    fc_resp.raise_for_status()
    fc_data = fc_resp.json()

    current = fc_data.get("current") or {}
    code = _safe_get(current, "weather_code")
    code_int = int(code) if code is not None else None
    text = WEATHER_CODE_ZH.get(code_int, "未知")

    return {
        "city": city,
        "latitude": lat,
        "longitude": lon,
        "timezone": str(fc_data.get("timezone", "auto")),
        "observation_time": str(_safe_get(current, "time", "")),
        "temperature_c": float(_safe_get(current, "temperature_2m", float("nan"))),
        "apparent_temperature_c": float(current["apparent_temperature"]) if "apparent_temperature" in current else None,
        "wind_speed_kmh": float(current["wind_speed_10m"]) if "wind_speed_10m" in current else None,
        "precipitation_mm": float(current["precipitation"]) if "precipitation" in current else None,
        "weather_code": code_int,
        "weather_text": text,
    }
