"""Unit tests for the Decorator-based feature pipeline."""

import pytest
from src.processing.features import (
    BaseFeatureExtractor,
    DrynessIndexDecorator,
    FeaturePipeline,
    FireWeatherIndexDecorator,
    GeoSample,
    TerrainRiskDecorator,
)


@pytest.fixture
def sample() -> GeoSample:
    return GeoSample(
        latitude=37.5, longitude=-119.5,
        ndvi=0.3, land_surface_temp=305.0,
        wind_speed=20.0, humidity=20.0,
        elevation=800.0, slope=15.0,
        days_since_rain=30, historical_fire=0,
    )


def test_base_extractor_returns_expected_keys(sample):
    features = BaseFeatureExtractor().extract(sample)
    assert "ndvi" in features
    assert "lst_celsius" in features
    assert abs(features["lst_celsius"] - (305.0 - 273.15)) < 0.01


def test_dryness_index_is_non_negative(sample):
    extractor = DrynessIndexDecorator(BaseFeatureExtractor())
    features = extractor.extract(sample)
    assert "dryness_index" in features
    assert features["dryness_index"] >= 0.0


def test_fwi_increases_with_wind(sample):
    extractor = FireWeatherIndexDecorator(BaseFeatureExtractor())
    low_wind = GeoSample(**{**sample.__dict__, "wind_speed": 5.0})
    high_wind = GeoSample(**{**sample.__dict__, "wind_speed": 35.0})
    assert extractor.extract(high_wind)["fire_weather_index"] > extractor.extract(low_wind)["fire_weather_index"]


def test_default_pipeline_returns_all_features(sample):
    pipeline = FeaturePipeline.default()
    features = pipeline.transform(sample)
    for key in ["ndvi", "dryness_index", "fire_weather_index", "terrain_risk"]:
        assert key in features, f"Missing feature: {key}"


def test_batch_transform_returns_dataframe(sample):
    import pandas as pd
    pipeline = FeaturePipeline.default()
    df = pipeline.transform_batch([sample, sample])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
