# Demo Guide

This is a 10 to 15 minute walkthrough script for presentation.

## 1. Start With The Problem

Explain that production ML models do not fail only because code breaks. They fail because the world changes. This is concept drift.

Key line:

> This project treats monitoring as part of the ML experiment, not just a dashboard afterthought.

## 2. Show The Architecture

Open `architecture/architecture.png`.

Explain the four planes:

- Inference
- Tracking
- Monitoring
- CI/CD

## 3. Start The Stack

```bash
docker compose up -d --build
```

Show:

```bash
docker compose ps
```

Expected services:

- api
- trainer
- drift_monitor
- mlflow
- postgres
- prometheus
- grafana

## 4. Show FastAPI

Open:

http://localhost:8000/docs

Then show:

```bash
curl http://localhost:8000/health
```

Run prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],"label":1}'
```

## 5. Show MLflow

Open:

http://localhost:5000

Explain:

- Trainer creates an experiment.
- Warmup run logs model parameters.
- Metrics include accuracy, F1, AUC, and log loss.
- Artifacts are stored in the MLflow artifact volume.

## 6. Show Prometheus

Open:

http://localhost:9090/targets

Explain target health:

- API metrics
- FastAPI request metrics
- Drift monitor metrics
- Prometheus self-scrape

Query:

```text
ml_predictions_total
```

## 7. Show Grafana

Open:

http://localhost:3000

Login:

```text
admin / admin
```

Dashboards:

- Model Performance
- Drift And Adaptive Retraining
- Infrastructure And API Health

## 8. Show The Code

Important files:

- `src/api/main.py`
- `src/drift/detectors.py`
- `src/monitoring/drift_service.py`
- `src/pipelines/experiment.py`
- `deploy/prometheus/prometheus.yml`
- `deploy/grafana/dashboards/`
- `.github/workflows/ci.yml`

## 9. Show The Research Paper

Open:

- `paper/main.tex`
- `paper/refs.bib`
- `experiments/results/results.csv`

Explain:

- HybridDD is the contribution.
- Benchmark compares six detectors.
- Metrics include quality, latency, throughput, delay, false positives, and retrains.

## 10. Close With Evaluation Mapping

Open:

`docs/RUBRIC_MAPPING.md`

End with:

> The project is not just a model. It is a reproducible MLOps system with tracking, deployment, monitoring, CI/CD, dashboards, research validation, and demo artifacts.

## Included Demo Artifacts

- Proper localhost video: `demo_artifacts/live_localhost_walkthrough.mp4`
- Smaller browser video: `demo_artifacts/live_localhost_walkthrough.webm`
- Screenshots: `demo_artifacts/screenshots/`
