import json
import requests
import time
from typing import Dict, Any, Optional

from src.utils.logger import get_logger, save_raw_payload, WEATHER_DIR, pipeline_run_id_var

logger = get_logger("weather_agent", "weather_agent.log")

class WeatherAgent:
    def __init__(self):
        self.geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
        self.weather_url = "https://api.open-meteo.com/v1/forecast"
        # Cache format: {cache_key: (timestamp, data)}
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = 900  # 15 minutes in seconds

    def _make_request_with_retry(self, url: str, params: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        backoff_factor = 1
        for attempt in range(max_retries):
            start_req = time.time()
            try:
                logger.info(f"API Request URL: {url} | Params: {params}")
                response = requests.get(url, params=params, timeout=10)
                req_time_ms = int((time.time() - start_req) * 1000)
                logger.info(f"API Response Status: {response.status_code} | Latency: {req_time_ms}ms")
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                req_time_ms = int((time.time() - start_req) * 1000)
                logger.error(f"API Request Failed in {req_time_ms}ms | URL: {url} | Error: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Weather API request failed after {max_retries} attempts: {e}")
                    raise
                wait_time = backoff_factor * (2 ** attempt)
                logger.warning(f"API request failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

    def get_weather(self, location_input: Optional[str] = None, lat: Optional[float] = None, lon: Optional[float] = None) -> Dict[str, Any]:
        """
        Retrieves weather data based on location string OR coordinates.
        """
        # Create a cache key based on inputs
        cache_key = f"{lat},{lon}" if lat is not None and lon is not None else str(location_input)
        
        # Check cache
        if cache_key in self.cache:
            timestamp, data = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                logger.info(f"Weather fetched from cache for key {cache_key}")
                return data

        try:
            logger.info(f"Getting weather for location: {location_input}, Lat: {lat}, Lon: {lon}")
            resolved_location = "Unknown Location"
            
            # Step 1: Geocoding if lat/lon not provided
            if lat is None or lon is None:
                if not location_input:
                    raise ValueError("Must provide either location_input or lat/lon coordinates.")
                
                geo_data = self._make_request_with_retry(self.geocode_url, {"name": location_input, "count": 1, "format": "json"})
                
                if "results" not in geo_data or len(geo_data["results"]) == 0:
                    raise ValueError(f"Could not resolve location: {location_input}")
                
                result = geo_data["results"][0]
                lat = result.get("latitude")
                lon = result.get("longitude")
                name = result.get("name", "")
                admin1 = result.get("admin1", "")
                country = result.get("country", "")
                
                parts = [p for p in [name, admin1, country] if p]
                resolved_location = ", ".join(parts)
            else:
                resolved_location = f"Lat: {lat}, Lon: {lon}"

            # Step 2: Fetch Weather Data
            weather_params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,showers,weather_code,cloud_cover,pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m",
                "timezone": "auto"
            }
            
            weather_data_resp = self._make_request_with_retry(self.weather_url, weather_params)
            
            current = weather_data_resp.get("current", {})
            
            response_data = {
                "status": "success",
                "location": resolved_location,
                "latitude": lat,
                "longitude": lon,
                "weather": {
                    "temperature": current.get("temperature_2m"),
                    "feels_like": current.get("apparent_temperature"),
                    "humidity": current.get("relative_humidity_2m"),
                    "precipitation": current.get("precipitation"),
                    "rain": current.get("rain"),
                    "cloud_cover": current.get("cloud_cover"),
                    "wind_speed": current.get("wind_speed_10m"),
                    "wind_direction": current.get("wind_direction_10m"),
                    "wind_gusts": current.get("wind_gusts_10m"),
                    "pressure": current.get("surface_pressure")
                }
            }
            
            logger.info(f"Weather data retrieved successfully for {resolved_location}")
            
            # Save raw payload
            run_id = pipeline_run_id_var.get()
            if run_id and run_id != "-":
                save_raw_payload(WEATHER_DIR, f"{run_id}.json", json.dumps(weather_data_resp, indent=2))
            
            # Save to cache
            self.cache[cache_key] = (time.time(), response_data)
            if location_input and lat is not None and lon is not None:
                self.cache[f"{lat},{lon}"] = (time.time(), response_data)
                
            return response_data

        except Exception as e:
            logger.error("WeatherAgent encountered an error.", exc_info=True)
            status = "timeout" if isinstance(e, requests.exceptions.Timeout) else "unavailable"
            return {
                "status": status,
                "error": "Weather service unavailable.",
                "details": str(e),
                "location": location_input if location_input else f"Lat: {lat}, Lon: {lon}",
                "latitude": lat,
                "longitude": lon,
                "weather": {
                    "temperature": None,
                    "feels_like": None,
                    "humidity": None,
                    "precipitation": None,
                    "rain": None,
                    "cloud_cover": None,
                    "wind_speed": None,
                    "wind_direction": None,
                    "wind_gusts": None,
                    "pressure": None
                }
            }
