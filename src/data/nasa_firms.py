"""
NASA FIRMS (Fire Information for Resource Management System) connector.
Fetches real historical fire detections from MODIS/VIIRS satellites.

Free API — get your key at: https://firms.modaps.eosdis.nasa.gov/api/area/
"""

from __future__ import annotations

import csv
import io
import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterator
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import URLError

from src.processing.features import GeoSample

logger = logging.getLogger(__name__)

# California bounding box — adjust for other regions
CALIFORNIA_BBOX = {
    "west": -124.48,
    "south": 32.53,
    "east": -114.13,
    "north": 42.01,
}


@dataclass
class FIRMSRecord:
    """Raw fire detection record from NASA FIRMS."""
    latitude: float
    longitude: float
    brightness: float       # Kelvin — proxy for fire intensity
    frp: float              # Fire Radiative Power (MW)
    confidence: str         # 'low', 'nominal', 'high'
    acq_date: str
    acq_time: str
    satellite: str          # 'Terra', 'Aqua', 'N' (NOAA-20), 'J1'


class NASAFIRMSClient:
    """
    Pulls fire detections from NASA FIRMS CSV API.

    Usage:
        client = NASAFIRMSClient(api_key="YOUR_KEY")
        for record in client.fetch_california(days=10):
            print(record.latitude, record.longitude, record.frp)

    Note: Without a key, use fetch_sample_data() which loads bundled CSVs.
    """

    BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

    def __init__(self, api_key: str | None = None) -> None:
        self._key = api_key

    def fetch_california(
        self,
        days: int = 10,
        source: str = "VIIRS_SNPP_NRT",
    ) -> Iterator[FIRMSRecord]:
        """
        Fetch recent fire detections over California.
        source options: VIIRS_SNPP_NRT, MODIS_NRT, VIIRS_NOAA20_NRT
        """
        if not self._key:
            logger.warning("No API key — using bundled sample data.")
            yield from self._load_sample()
            return

        bbox = f"{CALIFORNIA_BBOX['west']},{CALIFORNIA_BBOX['south']}," \
               f"{CALIFORNIA_BBOX['east']},{CALIFORNIA_BBOX['north']}"
        url = f"{self.BASE_URL}/{self._key}/{source}/{bbox}/{days}"

        logger.info("Fetching FIRMS data: %s", url)
        try:
            with urlopen(url, timeout=30) as resp:
                content = resp.read().decode("utf-8")
            yield from self._parse_csv(content)
        except URLError as e:
            logger.error("FIRMS API request failed: %s", e)
            logger.info("Falling back to sample data.")
            yield from self._load_sample()

    def _parse_csv(self, content: str) -> Iterator[FIRMSRecord]:
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            try:
                yield FIRMSRecord(
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    brightness=float(row.get("brightness", row.get("bright_ti4", 300))),
                    frp=float(row.get("frp", 0)),
                    confidence=str(row.get("confidence", "nominal")),
                    acq_date=row.get("acq_date", ""),
                    acq_time=row.get("acq_time", ""),
                    satellite=row.get("satellite", row.get("instrument", "unknown")),
                )
            except (ValueError, KeyError) as e:
                logger.debug("Skipping malformed FIRMS row: %s", e)

    def _load_sample(self) -> Iterator[FIRMSRecord]:
        """Load bundled Camp Fire 2018 sample data (no API key needed)."""
        import os
        sample_path = os.path.join(
            os.path.dirname(__file__), "../../data/sample/campfire_2018.csv"
        )
        if not os.path.exists(sample_path):
            logger.warning("Sample data not found — generating synthetic FIRMS records.")
            yield from _synthetic_firms_records()
            return
        with open(sample_path) as f:
            yield from self._parse_csv(f.read())


def _synthetic_firms_records(n: int = 500) -> Iterator[FIRMSRecord]:
    """
    Synthetic fire records centred on Paradise, CA (Camp Fire 2018 area).
    Used when neither API key nor sample CSV is available.
    """
    import random
    rng = random.Random(2018)
    # Paradise, CA: 39.7596° N, 121.6219° W
    for _ in range(n):
        yield FIRMSRecord(
            latitude=39.7596 + rng.gauss(0, 0.3),
            longitude=-121.6219 + rng.gauss(0, 0.3),
            brightness=rng.uniform(310, 420),
            frp=rng.uniform(5, 200),
            confidence=rng.choice(["nominal", "nominal", "high"]),
            acq_date="2018-11-08",
            acq_time=f"{rng.randint(0,23):02d}{rng.randint(0,59):02d}",
            satellite="Terra",
        )


# --------------------------------------------------------------------------- #
# Bridge: FIRMS record → GeoSample (for ML pipeline)
# --------------------------------------------------------------------------- #

def firms_to_geo_sample(
    record: FIRMSRecord,
    wind_speed: float = 10.0,
    humidity: float = 25.0,
    ndvi: float = 0.4,
    days_since_rain: int = 20,
) -> GeoSample:
    """
    Convert a FIRMS fire detection to a GeoSample.
    In production, wind/humidity/NDVI come from NOAA + Earth Engine lookups.
    For the retrospective notebook we inject historical NOAA values manually.
    """
    return GeoSample(
        latitude=record.latitude,
        longitude=record.longitude,
        ndvi=ndvi,
        land_surface_temp=record.brightness,   # FIRMS brightness ≈ LST proxy
        wind_speed=wind_speed,
        humidity=humidity,
        elevation=500.0,    # replace with USGS DEM lookup in production
        slope=10.0,
        days_since_rain=days_since_rain,
        historical_fire=1,  # it's a FIRMS record — fire was confirmed
    )
