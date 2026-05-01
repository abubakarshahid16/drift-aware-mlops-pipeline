# Architecture Guide

## Overview

This project is built as a production-style MLOps system rather than a single notebook. The pipeline separates training, inference, monitoring, tracking, and delivery into different services.

## Service Topology

| Service | Port | Role |
|---|---:|---|
| `api` | 8000 | FastAPI inference service |
| `trainer` | none | One-shot warmup training job |
| `drift_monitor` | 9100 | Streams labeled examples, detects drift, exposes drift metrics |
| `mlflow` | 5000 | Experiment tracking server |
| `postgres` | 5432 internal | MLflow backend store |
| `prometheus` | 9090 | Metrics scraper and query engine |
| `grafana` | 3000 | Dashboard UI |

## Data Flow

1. The trainer downloads and preprocesses ELEC2.
2. The trainer fits a warmup model and logs metrics/artifacts to MLflow.
3. The API loads a model from MLflow if available, otherwise from the local artifact fallback.
4. The drift monitor replays the processed stream into the API.
5. The API predicts and optionally updates the online model when labels are supplied.
6. The drift monitor updates rolling accuracy/error metrics and drift detector state.
7. Prometheus scrapes `/metrics` from the API and drift monitor.
8. Grafana dashboards visualize model quality, drift, retraining, and API health.
9. When drift is detected, the drift monitor calls `/reload` on the API.

## Important Design Choices

The API is deliberately separated from the drift monitor. This keeps inference simple and allows monitoring/adaptation logic to evolve independently.

The drift monitor has a `--hold` mode in Docker so the metrics endpoint remains alive after stream replay completes. This makes Grafana dashboards stable during demos.

The model registry loader checks MLflow reachability before calling the MLflow client. This avoids long local startup delays when MLflow is unavailable and keeps local tests fast.

## Architecture Files

- Mermaid source: `architecture/architecture.mmd`
- Dataflow source: `architecture/dataflow.mmd`
- Rendered diagram: `architecture/architecture.png`
