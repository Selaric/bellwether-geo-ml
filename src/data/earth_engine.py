"""
Google Earth Engine (GEE) and NOAA connectors.

GEE gives you MODIS NDVI, Land Surface Temperature, and Landsat imagery.
NOAA gives you wind, humidity, precipitation.

Setup (one-time):
    pip install earthengine-api
    earthengine authenticate          # opens browser OAuth
    # OR use a service account JSON for production

NOAA API is free, no key needed for basic endpoints.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.request import urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Google Earth Engine — NDVI + LST
# --------------------------------------------------------------------------- #

class EarthEngineClient:
    """
    Fetches MODIS-derived NDVI and Land Surface Temperature for a lat/lon.

    Requires: pip install earthengine-api
    Auth:      earthengine authenticate   (or service account in prod)

    If EE is unavailable, falls back to synthetic values so the rest of
    the pipeline keeps working.
    """

    def __init__(self) -> None:
        self._initialized = False
        self._try_init()

    def _try_init(self) -> None:
        try:
            import ee
            ee.Initialize()
            self._ee = ee
            self._initialized = True
            logger.info("Google Earth Engine initialized ✓")
        except Exception as e:
            logger.warning("Earth Engine unavailable (%s) — using fallback values.", e)

    def get_ndvi(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
    ) -> float:
        """
        Returns mean NDVI over a 500m radius for the given date range.
        Uses MODIS MOD13A1 (500m, 16-day composite).
        """
        if not self._initialized:
            return self._fallback_ndvi(lat, lon)

        ee = self._ee
        point = ee.Geometry.Point([lon, lat])
        collection = (
            ee.ImageCollection("MODIS/006/MOD13A1")
            .filterDate(start_date, end_date)
            .select("NDVI")
        )
        mean_image = collection.mean()
        value = mean_image.sample(point, 500).first().get("NDVI").getInfo()
        # MODIS NDVI is scaled by 0.0001
        return float(value) * 0.0001 if value is not None else self._fallback_ndvi(lat, lon)

    def get_land_surface_temp(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
    ) -> float:
        """
        Returns mean LST (Kelvin) from MODIS MOD11A2 (1km, 8-day).
        """
        if not self._initialized:
            return 305.0  # ~32°C fallback

        ee = self._ee
        point = ee.Geometry.Point([lon, lat])
        collection = (
            ee.ImageCollection("MODIS/006/MOD11A2")
            .filterDate(start_date, end_date)
            .select("LST_Day_1km")
        )
        mean_image = collection.mean()
        value = mean_image.sample(point, 1000).first().get("LST_Day_1km").getInfo()
        # MODIS LST scale factor is 0.02
        return float(value) * 0.02 if value is not None else 305.0

    def _fallback_ndvi(self, lat: float, lon: float) -> float:
        """Rough synthetic NDVI based on California latitude bands."""
        if lat > 40:
            return 0.6   # Northern CA — wetter, greener
        elif lat > 37:
            return 0.35  # Central CA
        else:
            return 0.15  # Southern CA — drier


# --------------------------------------------------------------------------- #
# NOAA Weather API
# --------------------------------------------------------------------------- #

@dataclass
class NOAAObservation:
    wind_speed: float        # m/s
    wind_direction: float    # degrees
    humidity: float          # %
    temperature: float       # Celsius
    precip_last_hour: float  # mm


class NOAAWeatherClient:
    """
    Fetches current/historical weather from NOAA's free public API.
    No API key required.

    Docs: https://www.weather.gov/documentation/services-web-api
    """

    POINTS_URL = "https://api.weather.gov/points/{lat},{lon}"
    STATIONS_URL = "https://api.weather.gov/points/{lat},{lon}/stations"
    OBS_URL = "https://api.weather.gov/stations/{station}/observations/latest"

    def get_observation(self, lat: float, lon: float) -> NOAAObservation | None:
        """Get the latest observation for the nearest NOAA station."""
        try:
            station = self._nearest_station(lat, lon)
            if not station:
                return None
            url = self.OBS_URL.format(station=station)
            data = self._fetch_json(url)
            props = data["properties"]
            return NOAAObservation(
                wind_speed=self._value(props, "windSpeed") or 0.0,
                wind_direction=self._value(props, "windDirection") or 0.0,
                humidity=self._value(props, "relativeHumidity") or 50.0,
                temperature=self._value(props, "temperature") or 20.0,
                precip_last_hour=self._value(props, "precipitationLastHour") or 0.0,
            )
        except Exception as e:
            logger.warning("NOAA API failed for (%.4f, %.4f): %s", lat, lon, e)
            return None

    def _nearest_station(self, lat: float, lon: float) -> str | None:
        url = self.STATIONS_URL.format(lat=round(lat, 4), lon=round(lon, 4))
        try:
            data = self._fetch_json(url)
            features = data.get("features", [])
            if features:
                return features[0]["properties"]["stationIdentifier"]
        except Exception as e:
            logger.debug("Station lookup failed: %s", e)
        return None

    def _fetch_json(self, url: str) -> dict[str, Any]:
        from urllib.request import Request
        req = Request(url, headers={"User-Agent": "BellwetherML/1.0 (research)"})
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    @staticmethod
    def _value(props: dict, key: str) -> float | None:
        v = props.get(key, {})
        return v.get("value") if isinstance(v, dict) else None


# --------------------------------------------------------------------------- #
# Convenience: enrich a lat/lon with all real data sources
# --------------------------------------------------------------------------- #

def enrich_location(
    lat: float,
    lon: float,
    start_date: str = "2018-10-01",
    end_date: str = "2018-11-07",
    ee_client: EarthEngineClient | None = None,
    noaa_client: NOAAWeatherClient | None = None,
) -> dict[str, float]:
    """
    Pulls NDVI + LST from Earth Engine and weather from NOAA for a location.
    Returns a dict that maps directly to GeoSample fields.
    """
    ee = ee_client or EarthEngineClient()
    noaa = noaa_client or NOAAWeatherClient()

    ndvi = ee.get_ndvi(lat, lon, start_date, end_date)
    lst = ee.get_land_surface_temp(lat, lon, start_date, end_date)
    obs = noaa.get_observation(lat, lon)

    return {
        "latitude": lat,
        "longitude": lon,
        "ndvi": ndvi,
        "land_surface_temp": lst,
        "wind_speed": obs.wind_speed if obs else 15.0,
        "humidity": obs.humidity if obs else 20.0,
        "elevation": 500.0,   # enrich with USGS DEM for full production
        "slope": 10.0,
        "days_since_rain": 25,
        "historical_fire": 0,
    }
