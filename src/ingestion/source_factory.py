"""
Factory Pattern — Instantiate geospatial data sources by name.
Isolates construction logic so the pipeline never deals with source details.
"""

from __future__ import annotations

import csv
import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Iterator

from src.processing.features import GeoSample

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract data source. All sources yield GeoSamples."""

    @abstractmethod
    def stream(self) -> Iterator[GeoSample]:
        ...


class MockDataSource(DataSource):
    """
    Synthetic geospatial data — useful for unit tests and CI pipelines
    where live API/satellite data is unavailable.
    """

    def __init__(self, n_samples: int = 1000, seed: int = 42) -> None:
        self._n = n_samples
        self._rng = random.Random(seed)

    def stream(self) -> Iterator[GeoSample]:
        rng = self._rng
        for _ in range(self._n):
            yield GeoSample(
                latitude=rng.uniform(32.0, 42.0),
                longitude=rng.uniform(-124.0, -114.0),
                ndvi=rng.uniform(-0.1, 0.9),
                land_surface_temp=rng.uniform(290.0, 330.0),
                wind_speed=rng.uniform(0.0, 40.0),
                humidity=rng.uniform(5.0, 95.0),
                elevation=rng.uniform(0.0, 3500.0),
                slope=rng.uniform(0.0, 45.0),
                days_since_rain=rng.randint(0, 90),
                historical_fire=rng.choice([0, 0, 0, 1]),
            )


class CSVDataSource(DataSource):
    """Streams GeoSamples from a CSV file. Low memory — reads row by row."""

    REQUIRED_COLUMNS = {
        "latitude", "longitude", "ndvi", "land_surface_temp",
        "wind_speed", "humidity", "elevation", "slope",
        "days_since_rain", "historical_fire",
    }

    def __init__(self, path: Path) -> None:
        self._path = path

    def stream(self) -> Iterator[GeoSample]:
        with open(self._path, newline="") as f:
            reader = csv.DictReader(f)
            missing = self.REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV missing columns: {missing}")
            for i, row in enumerate(reader):
                try:
                    yield GeoSample(**{k: float(row[k]) if k != "historical_fire" and k != "days_since_rain"
                                       else int(float(row[k])) for k in self.REQUIRED_COLUMNS})
                except (ValueError, KeyError) as exc:
                    logger.warning("Skipping row %d: %s", i, exc)


class SourceFactory:
    """
    Creates DataSource instances by name.
    Register new sources at runtime without modifying existing code.
    """

    _registry: dict[str, type[DataSource]] = {
        "mock": MockDataSource,
        "csv": CSVDataSource,
    }

    @classmethod
    def register(cls, name: str, source_cls: type[DataSource]) -> None:
        cls._registry[name] = source_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> DataSource:
        if name not in cls._registry:
            raise ValueError(f"Unknown source '{name}'. Available: {list(cls._registry)}")
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry.keys())
