import json
import re
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
        self.location_cache: Dict[str, Dict[str, Any]] = {}
        self.preferred_country = "IN"

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

    def _geocode_location(self, location_input: str) -> Dict[str, Any]:
        """
        Geocodes a user location query string into coordinates and display details.
        
        Attempts:
        1. Nominatim + countrycodes=in
        2. Nominatim global search
        3. Open-Meteo geocoder
        """
        # Input cleaning & normalization
        cleaned_input = location_input.strip()
        cleaned_input = re.sub(r"\s+", " ", cleaned_input)
        
        cache_key = cleaned_input.lower()
        if cache_key in self.location_cache:
            logger.info(f"Location resolution cache hit for '{cleaned_input}'")
            return self.location_cache[cache_key]

        start_time = time.time()
        
        def parse_nominatim_result(result: Dict[str, Any], source: str) -> Dict[str, Any]:
            address = result.get("address", {})
            name = result.get("name", "")
            country = address.get("country", "")
            state = address.get("state", address.get("state_district", ""))
            
            town = address.get("town", address.get("city", address.get("village", address.get("hamlet", address.get("suburb")))))
            county = address.get("county")
            
            parts = [p for p in [name, town, county, state, country] if p]
            if not parts:
                resolved_location = result.get("display_name", "Unknown")
            else:
                seen = set()
                dedup_parts = []
                for p in parts:
                    if p not in seen:
                        seen.add(p)
                        dedup_parts.append(p)
                resolved_location = ", ".join(dedup_parts)
                
            return {
                "latitude": float(result["lat"]),
                "longitude": float(result["lon"]),
                "resolved_location": resolved_location,
                "country": country,
                "state": state,
                "source": source
            }

        # Attempt 1: Nominatim + countrycodes=in
        logger.info(f"Geocoder Attempt 1: Nominatim (India) | Query: '{cleaned_input}'")
        nominatim_url = "https://nominatim.openstreetmap.org/search"
        headers = {"User-Agent": "CropGuardianAI/1.1"}
        
        try:
            response = requests.get(
                nominatim_url,
                headers=headers,
                params={
                    "q": cleaned_input,
                    "format": "jsonv2",
                    "limit": 1,
                    "addressdetails": 1,
                    "countrycodes": self.preferred_country.lower()
                },
                timeout=10
            )
            response.raise_for_status()
            results = response.json()
            if results:
                india_res = results[0]
                india_rank = india_res.get("place_rank", 30)
                
                # If it's a street, shop, or POI (rank >= 25), check if global search has a more prominent match (lower rank)
                if india_rank >= 25:
                    logger.info(f"India match for '{cleaned_input}' is a street/POI (rank {india_rank} >= 25). Checking global match...")
                    try:
                        global_resp = requests.get(
                            nominatim_url,
                            headers=headers,
                            params={
                                "q": cleaned_input,
                                "format": "jsonv2",
                                "limit": 1,
                                "addressdetails": 1
                            },
                            timeout=10
                        )
                        global_resp.raise_for_status()
                        global_results = global_resp.json()
                        if global_results:
                            global_res = global_results[0]
                            global_rank = global_res.get("place_rank", 30)
                            if global_rank < india_rank:
                                logger.info(f"Global match for '{cleaned_input}' is more prominent (rank {global_rank} < {india_rank}). Using global match.")
                                res_data = parse_nominatim_result(global_res, "nominatim")
                                
                                # Check for country warning
                                country_code = global_res.get("address", {}).get("country_code", "")
                                if country_code.upper() != self.preferred_country.upper():
                                    logger.warning(f"Suspicious resolution outside primary country ({self.preferred_country}): Resolved country is '{res_data['country']}'")
                                
                                resolution_time = time.time() - start_time
                                logger.info(
                                    f"Geocoder: Nominatim\n"
                                    f"Query: {cleaned_input}\n"
                                    f"Resolved: {res_data['resolved_location']}\n"
                                    f"Lat: {res_data['latitude']}\n"
                                    f"Lon: {res_data['longitude']}\n"
                                    f"Country: {res_data['country']}\n"
                                    f"Status: SUCCESS\n"
                                    f"Resolution Time: {resolution_time:.2f}s"
                                )
                                self.location_cache[cache_key] = res_data
                                return res_data
                    except Exception as ge:
                        logger.error(f"Global check failed for query '{cleaned_input}': {ge}", exc_info=True)
                
                # Use the India result
                res_data = parse_nominatim_result(india_res, "nominatim")
                resolution_time = time.time() - start_time
                logger.info(
                    f"Geocoder: Nominatim\n"
                    f"Query: {cleaned_input}\n"
                    f"Resolved: {res_data['resolved_location']}\n"
                    f"Lat: {res_data['latitude']}\n"
                    f"Lon: {res_data['longitude']}\n"
                    f"Country: {res_data['country']}\n"
                    f"Status: SUCCESS\n"
                    f"Resolution Time: {resolution_time:.2f}s"
                )
                self.location_cache[cache_key] = res_data
                return res_data
            else:
                logger.warning(f"Geocoder Attempt 1: Nominatim (India) | Query: '{cleaned_input}' | Status: FAILED | Reason: No results returned")
        except Exception as e:
            logger.error(f"Geocoder Attempt 1: Nominatim (India) | Query: '{cleaned_input}' | Status: FAILED | Error: {e}", exc_info=True)

        # Attempt 2: Nominatim global search
        logger.info(f"Geocoder Attempt 2: Nominatim (Global) | Query: '{cleaned_input}'")
        try:
            response = requests.get(
                nominatim_url,
                headers=headers,
                params={
                    "q": cleaned_input,
                    "format": "jsonv2",
                    "limit": 1,
                    "addressdetails": 1
                },
                timeout=10
            )
            response.raise_for_status()
            results = response.json()
            if results:
                res_data = parse_nominatim_result(results[0], "nominatim")
                resolution_time = time.time() - start_time
                
                # Check for country warning
                country_code = results[0].get("address", {}).get("country_code", "")
                if country_code.upper() != self.preferred_country.upper():
                    logger.warning(f"Suspicious resolution outside primary country ({self.preferred_country}): Resolved country is '{res_data['country']}'")
                
                logger.info(
                    f"Geocoder: Nominatim (Global)\n"
                    f"Query: {cleaned_input}\n"
                    f"Resolved: {res_data['resolved_location']}\n"
                    f"Lat: {res_data['latitude']}\n"
                    f"Lon: {res_data['longitude']}\n"
                    f"Country: {res_data['country']}\n"
                    f"Status: SUCCESS\n"
                    f"Resolution Time: {resolution_time:.2f}s"
                )
                self.location_cache[cache_key] = res_data
                return res_data
            else:
                logger.warning(f"Geocoder Attempt 2: Nominatim (Global) | Query: '{cleaned_input}' | Status: FAILED | Reason: No results returned")
        except Exception as e:
            logger.error(f"Geocoder Attempt 2: Nominatim (Global) | Query: '{cleaned_input}' | Status: FAILED | Error: {e}", exc_info=True)

        # Attempt 3: Open-Meteo geocoder
        logger.info(f"Geocoder Attempt 3: Open-Meteo | Query: '{cleaned_input}'")
        try:
            geo_data = self._make_request_with_retry(
                self.geocode_url, 
                {"name": cleaned_input, "count": 1, "format": "json"}
            )
            if "results" in geo_data and len(geo_data["results"]) > 0:
                result = geo_data["results"][0]
                lat = float(result.get("latitude"))
                lon = float(result.get("longitude"))
                name = result.get("name", "")
                admin1 = result.get("admin1", "")
                country = result.get("country", "")
                country_code = result.get("country_code", "")
                
                parts = [p for p in [name, admin1, country] if p]
                resolved_location = ", ".join(parts)
                
                res_data = {
                    "latitude": lat,
                    "longitude": lon,
                    "resolved_location": resolved_location,
                    "country": country,
                    "state": admin1,
                    "source": "open-meteo"
                }
                
                # Check for country warning
                if country_code.upper() != self.preferred_country.upper():
                    logger.warning(f"Suspicious resolution outside primary country ({self.preferred_country}): Resolved country is '{res_data['country']}'")
                
                resolution_time = time.time() - start_time
                logger.info(
                    f"Geocoder: Open-Meteo\n"
                    f"Query: {cleaned_input}\n"
                    f"Resolved: {res_data['resolved_location']}\n"
                    f"Lat: {res_data['latitude']}\n"
                    f"Lon: {res_data['longitude']}\n"
                    f"Country: {res_data['country']}\n"
                    f"Status: SUCCESS\n"
                    f"Resolution Time: {resolution_time:.2f}s"
                )
                self.location_cache[cache_key] = res_data
                return res_data
            else:
                logger.warning(f"Geocoder Attempt 3: Open-Meteo | Query: '{cleaned_input}' | Status: FAILED | Reason: No results returned")
        except Exception as e:
            logger.error(f"Geocoder Attempt 3: Open-Meteo | Query: '{cleaned_input}' | Status: FAILED | Error: {e}", exc_info=True)

        # If all failed, raise ValueError
        resolution_time = time.time() - start_time
        logger.error(f"Geocoder all attempts failed | Query: {cleaned_input} | Status: FAILED | Resolution Time: {resolution_time:.2f}s")
        raise ValueError(f"Could not resolve location after trying all geocoder services: {location_input}")

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
            source = "coordinates" if lat is not None and lon is not None else "unknown"
            
            # Step 1: Geocoding if lat/lon not provided
            if lat is None or lon is None:
                if not location_input:
                    raise ValueError("Must provide either location_input or lat/lon coordinates.")
                
                geo_res = self._geocode_location(location_input)
                lat = geo_res["latitude"]
                lon = geo_res["longitude"]
                resolved_location = geo_res["resolved_location"]
                source = geo_res["source"]
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
                "location_resolution": {
                    "input": location_input if location_input else f"{lat},{lon}",
                    "resolved_name": resolved_location,
                    "latitude": lat,
                    "longitude": lon,
                    "source": source
                },
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
                "location_resolution": {
                    "input": location_input if location_input else (f"{lat},{lon}" if lat is not None and lon is not None else "unknown"),
                    "resolved_name": resolved_location,
                    "latitude": lat,
                    "longitude": lon,
                    "source": source
                },
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
