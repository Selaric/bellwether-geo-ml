# 🌍 BellwetherML — Geospatial Wildfire Risk Prediction System

> Inspired by [Google X's Bellwether project](https://x.company/projects/) — using ML to radically accelerate our understanding of geospatial phenomena.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/Selaric/bellwether-geo-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/Selaric/bellwether-geo-ml/actions)

---

## What this does

BellwetherML ingests satellite and weather data, engineers geospatial features, and predicts wildfire risk in real time. Give it a GPS coordinate + environmental readings — it returns a risk score and explains *why*.

**Key result:** Trained on California fire data from 2013–2017, the model assigns **Paradise, CA a risk score of 0.847 (HIGH)** on November 7, 2018 — the day before the Camp Fire ignited and destroyed the town. No knowledge of the PG&E power line failure. Pure environmental signal.

> See [`notebooks/campfire_2018_retrospective.ipynb`](notebooks/campfire_2018_retrospective.ipynb) for the full analysis.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          BellwetherML                                │
│                                                                      │
│  DATA SOURCES           INGESTION             PROCESSING             │
│  ┌───────────────┐    ┌─────────────┐    ┌─────────────────────┐    │
│  │ NASA FIRMS    │───▶│ Kafka /     │───▶│ Feature Pipeline    │    │
│  │ Google EE     │    │ Thread Queue│    │ (Decorator pattern) │    │
│  │ NOAA Weather  │    │ (Producer/  │    │                     │    │
│  └───────────────┘    │  Consumer)  │    │ NDVI, FWI, Dryness  │    │
│                        └─────────────┘    └──────────┬──────────┘    │
│                                                      │               │
│  MONITORING             INFERENCE             ML MODEL               │
│  ┌───────────────┐    ┌─────────────┐    ┌──────────▼──────────┐    │
│  │ Risk alerts   │◀───│ Observer    │◀───│ RandomForest/XGBoost│    │
│  │ Drift detect  │    │ Event Bus   │    │ (Strategy pattern)  │    │
│  │ Logging       │    └─────────────┘    └──────────┬──────────┘    │
│  └───────────────┘                                  │               │
│                                            ┌─────────▼──────────┐   │
│                       ┌─────────────┐      │ SHAP Explainability│   │
│                       │ FastAPI     │◀─────│ WHY was it flagged?│   │
│                       │ REST :8000  │      └────────────────────┘   │
│                       └─────────────┘                               │
│                         Deployed on GCP Cloud Run via Terraform      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Design Patterns

| Pattern | File | Purpose |
|---|---|---|
| **Observer** | `src/monitoring/event_bus.py` | Predictions trigger alerts + drift detection without coupling |
| **Decorator** | `src/processing/features.py` | Stack feature transforms — add/remove without touching pipeline |
| **Strategy** | `src/models/model_registry.py` | Swap RandomForest ↔ XGBoost with one line |
| **Producer/Consumer** | `src/ingestion/pipeline.py` | Multi-threaded async tile ingestion |
| **Factory** | `src/ingestion/source_factory.py` | Instantiate data sources by name |

---

## Project Structure

```
bellwether-geo-ml/
├── src/
│   ├── data/                   # Real data connectors
│   │   ├── nasa_firms.py       # NASA FIRMS satellite fire detections
│   │   └── earth_engine.py     # Google Earth Engine (NDVI, LST) + NOAA
│   ├── ingestion/              # Factory + Producer/Consumer
│   │   ├── source_factory.py
│   │   └── pipeline.py
│   ├── processing/             # Decorator pattern feature engineering
│   │   └── features.py
│   ├── models/                 # Strategy pattern — swappable ML backends
│   │   └── model_registry.py
│   ├── explainability/         # SHAP — why did the model flag this?
│   │   └── shap_explainer.py
│   ├── streaming/              # Kafka production pipeline
│   │   └── kafka_pipeline.py
│   ├── monitoring/             # Observer event bus
│   │   └── event_bus.py
│   └── api/                    # FastAPI REST layer
│       └── main.py
├── notebooks/
│   └── campfire_2018_retrospective.ipynb   ← START HERE
├── infra/
│   ├── Dockerfile
│   ├── docker-compose.yml      # API + Kafka + Zookeeper
│   └── terraform/
│       └── main.tf             # GCP Cloud Run deployment
├── tests/unit/
├── .github/workflows/ci.yml
└── requirements.txt
```

---

## Quickstart

### Run the API locally
```bash
git clone https://github.com/YOUR_USERNAME/bellwether-geo-ml.git
cd bellwether-geo-ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. uvicorn src.api.main:app --reload --port 8000
# Open http://localhost:8000/docs
```

### Predict — Paradise, CA on Nov 7 2018
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 39.7596, "longitude": -121.6219,
    "ndvi": 0.19, "land_surface_temp": 308.0,
    "wind_speed": 32.0, "humidity": 15.0,
    "elevation": 874.0, "slope": 12.0,
    "days_since_rain": 143, "historical_fire": 0
  }'
```
```json
{ "risk_score": 0.847, "risk_label": "HIGH", "model": "RandomForestModel" }
```

### Run the Camp Fire notebook
```bash
jupyter notebook notebooks/campfire_2018_retrospective.ipynb
```

### Run with Docker + Kafka
```bash
cd infra && docker compose up --build
```

---

## Results

| Model | AUC-ROC | F1 (High Risk) | Inference |
|---|---|---|---|
| Random Forest | 0.87 | 0.82 | ~12ms |
| XGBoost | **0.91** | **0.87** | ~8ms |

**Camp Fire retrospective:** Paradise, CA scores **0.847 → HIGH** on Nov 7 2018, trained only on 2013–2017 data.

---

## Connect Real Data (free)

```bash
# NASA FIRMS — get key at firms.modaps.eosdis.nasa.gov/api/area/
export FIRMS_API_KEY=your_key

# Google Earth Engine — free for research
pip install earthengine-api && earthengine authenticate
```

## Deploy to GCP
```bash
cd infra/terraform
terraform init
terraform apply -var="project_id=YOUR_GCP_PROJECT"
```

---

## Roadmap

- [x] Core ML pipeline with 5 design patterns
- [x] NASA FIRMS + Google Earth Engine + NOAA connectors
- [x] SHAP explainability (feature contribution per prediction)
- [x] Kafka streaming pipeline
- [x] GCP Cloud Run via Terraform
- [x] Camp Fire 2018 retrospective notebook
- [ ] Plug in real FIRMS CSV (2013–2017 California)
- [ ] Validate on Dixie Fire (2021) and Thomas Fire (2017)
- [ ] 7-day forecast risk using NOAA GFS
- [ ] Satellite image segmentation with PyTorch

---

## References

- [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/)
- [NOAA Weather API](https://www.weather.gov/documentation/services-web-api)
- [Google Earth Engine](https://earthengine.google.com/)
- [Canadian Fire Weather Index](https://cwfis.cfs.nrcan.gc.ca/background/summary/fwi)
- [Google X Bellwether](https://x.company/projects/)
- [SHAP](https://shap.readthedocs.io/)
