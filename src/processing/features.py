"""
Decorator Pattern — Composable, stackable feature engineering transforms.
Each decorator wraps a base feature extractor, adding a new derived feature.
This mirrors production ML pipelines where features are added incrementally.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GeoSample:
    """Raw geospatial observation before feature extraction."""
    latitude: float
    longitude: float
    ndvi: float                  # Normalized Difference Vegetation Index [-1, 1]
    land_surface_temp: float     # Kelvin
    wind_speed: float            # m/s
    humidity: float              # % relative humidity
    elevation: float             # meters
    slope: float                 # degrees
    days_since_rain: int
    historical_fire: int         # 1 if fire recorded in past 5 years

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class FeatureExtractor(ABC):
    """Base component in the Decorator chain."""

    @abstractmethod
    def extract(self, sample: GeoSample) -> dict[str, float]:
        ...


class BaseFeatureExtractor(FeatureExtractor):
    """Concrete base — passes raw scalar features through."""

    def extract(self, sample: GeoSample) -> dict[str, float]:
        return {
            "ndvi": sample.ndvi,
            "lst_celsius": sample.land_surface_temp - 273.15,
            "wind_speed": sample.wind_speed,
            "humidity": sample.humidity,
            "elevation": sample.elevation,
            "slope": sample.slope,
            "days_since_rain": float(sample.days_since_rain),
            "historical_fire": float(sample.historical_fire),
        }


class FeatureDecorator(FeatureExtractor, ABC):
    """Abstract decorator — wraps another extractor and augments its output."""

    def __init__(self, wrapped: FeatureExtractor) -> None:
        self._wrapped = wrapped

    def extract(self, sample: GeoSample) -> dict[str, float]:
        features = self._wrapped.extract(sample)
        return self._augment(features, sample)

    @abstractmethod
    def _augment(self, features: dict[str, float], sample: GeoSample) -> dict[str, float]:
        ...


class DrynessIndexDecorator(FeatureDecorator):
    """Adds a composite dryness index: combines NDVI, humidity, and rain lag."""

    def _augment(self, features: dict[str, float], sample: GeoSample) -> dict[str, float]:
        # Higher = drier = higher fire risk
        dryness = (1 - sample.ndvi) * (1 - sample.humidity / 100) * np.log1p(sample.days_since_rain)
        features["dryness_index"] = float(np.clip(dryness, 0, 10))
        return features


class FireWeatherIndexDecorator(FeatureDecorator):
    """
    Simplified Fire Weather Index (FWI) — based on Canadian FWI system.
    Combines temperature, wind, and humidity into a single risk score.
    """

    def _augment(self, features: dict[str, float], sample: GeoSample) -> dict[str, float]:
        temp = features.get("lst_celsius", 20.0)
        wind = sample.wind_speed
        rh = sample.humidity
        fwi = (temp * wind) / max(rh, 1.0)
        features["fire_weather_index"] = float(np.clip(fwi, 0, 500))
        return features


class TerrainRiskDecorator(FeatureDecorator):
    """Adds terrain-based risk: steep south-facing slopes dry out faster."""

    def _augment(self, features: dict[str, float], sample: GeoSample) -> dict[str, float]:
        # Steeper slopes + higher elevation = higher terrain risk
        terrain_risk = (sample.slope / 90.0) * np.log1p(sample.elevation / 1000.0)
        features["terrain_risk"] = float(np.clip(terrain_risk, 0, 5))
        return features


class FeaturePipeline:
    """
    Builds and applies a decorator chain of feature extractors.
    Usage:
        pipeline = FeaturePipeline.default()
        features = pipeline.transform(sample)
    """

    def __init__(self, extractor: FeatureExtractor) -> None:
        self._extractor = extractor

    @classmethod
    def default(cls) -> "FeaturePipeline":
        """Constructs the standard production feature stack."""
        extractor: FeatureExtractor = BaseFeatureExtractor()
        extractor = DrynessIndexDecorator(extractor)
        extractor = FireWeatherIndexDecorator(extractor)
        extractor = TerrainRiskDecorator(extractor)
        return cls(extractor)

    def transform(self, sample: GeoSample) -> dict[str, float]:
        features = self._extractor.extract(sample)
        logger.debug("Extracted %d features for (%.4f, %.4f)", len(features),
                     sample.latitude, sample.longitude)
        return features

    def transform_batch(self, samples: list[GeoSample]) -> pd.DataFrame:
        rows = [self.transform(s) for s in samples]
        return pd.DataFrame(rows)
