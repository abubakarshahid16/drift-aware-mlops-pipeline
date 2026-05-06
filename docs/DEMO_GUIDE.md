# Demo Guide

This is a 15 minute defense script for the final presentation deck:

```text
presentation/Drift-Aware-MLOps-Final-Defense.pptx
```

The goal is to make the examiner's grading job easy: every spoken section maps
to one or more rubric criteria and one or more repository artifacts.

## 15 Minute Run Of Show

| Time | Slides | What to prove | Rubric coverage |
|---:|---|---|---|
| 0:00-2:00 | 1-3 | The problem is real and the submission is evidence-first. | Research novelty, presentation |
| 2:00-4:00 | 4-5 | HybridDD is the research contribution. | Research novelty |
| 4:00-6:30 | 6-7 | The system is a complete MLOps implementation. | Technical implementation, design |
| 6:30-9:00 | 8 | Monitoring is operational and research-relevant. | Monitoring and observability |
| 9:00-12:00 | 9-10 | Experiments are reproducible and statistically analyzed. | Experimental rigor |
| 12:00-14:00 | 11 | Live demo follows the exact evidence path. | Technical implementation, monitoring |
| 14:00-15:00 | 12 | Close on the 100/100 rubric map and Q&A anchors. | Documentation and presentation |

## Slide Talk Track

1. Cover: "This is a complete drift-aware MLOps pipeline, not only a model."
2. Rubric scorecard: "Each weighted category has direct evidence in the repo."
3. Problem: "Models can be operationally healthy while statistically stale."
4. Literature: "Single-signal detectors have different blind spots."
5. HybridDD: "The contribution combines DDM and KSWIN with consensus, override,
   and cooldown logic."
6. Architecture: "Inference, tracking, monitoring, and CI/CD are separated."
7. Implementation: "Every mandatory tool is integrated and executable."
8. Observability: "Prometheus and Grafana expose model quality, drift, latency,
   retraining, and infrastructure health."
9. Experiments: "The harness uses prequential evaluation across detectors,
   streams, and seeds."
10. Results: "The refreshed compact benchmark has applicable Friedman/Nemenyi
    tests for accuracy, penalized delay, and false positives."
11. Demo plan: "Now I will show the same path live."
12. Close: "The project is reproducible, observable, documented, and research-backed."

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
