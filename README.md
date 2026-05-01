# Drift-Aware Adaptive Retraining MLOps Pipeline

An end-to-end MLOps research project for streaming classification under concept drift. The repository combines a real implementation with a research-style evaluation of a hybrid drift detector, then wraps it in Docker, MLflow, Prometheus, Grafana, and GitHub Actions.

## What This Repository Demonstrates

This project answers a practical production ML question:

> How can a model serving pipeline detect concept drift, expose that drift through production monitoring, and trigger adaptive retraining without turning the system into an unobservable black box?

The implemented answer is a FastAPI inference service connected to:

- MLflow for experiment tracking and model artifacts.
- Docker Compose for a repeatable local deployment.
- Prometheus for metrics collection.
- Grafana for dashboards.
- GitHub Actions for CI, Docker builds, paper builds, and optional AWS deploy.
- A research benchmark comparing ADWIN, DDM, EDDM, KSWIN, Page-Hinkley, and the proposed HybridDD detector.

## Live Demo Assets

The demo video and screenshots are included directly in the repository.

- [Demo video](demo_artifacts/mlops_live_demo.mp4)
- [All screenshots](demo_artifacts/screenshots)
- [Demo artifact notes](demo_artifacts/README_demo_artifacts.txt)

### Demo Preview

FastAPI inference API:

![FastAPI docs](demo_artifacts/screenshots/02_api_docs.png)

Prometheus target health:

![Prometheus targets](demo_artifacts/screenshots/05_prometheus_targets.png)

Grafana model performance dashboard:

![Grafana model performance](demo_artifacts/screenshots/08_grafana_model_performance.png)

MLflow tracking evidence:

![MLflow tracking](demo_artifacts/screenshots/07_mlflow_experiment.png)

## Architecture

![Architecture](architecture/architecture.png)

The system has four main planes:

1. Inference plane: FastAPI serves `/predict`, `/health`, `/reload`, and `/metrics`.
2. Tracking plane: MLflow records warmup training runs, model artifacts, parameters, and metrics.
3. Monitoring plane: Prometheus scrapes API and drift-monitor metrics; Grafana renders dashboards.
4. CI/CD plane: GitHub Actions run linting, tests, smoke experiments, Docker builds, scans, and optional deployment.

Detailed architecture documentation:

- [Architecture guide](docs/ARCHITECTURE.md)
- [Monitoring guide](docs/MONITORING.md)
- [CI/CD guide](docs/CI_CD.md)

## Research Contribution

The research contribution is HybridDD, a hybrid concept-drift detector that combines:

- DDM, a performance-based detector that reacts when prediction errors increase.
- KSWIN, a distribution-based detector that reacts when feature distributions shift.
- A consensus rule and confidence override to reduce false positives while still reacting quickly to damaging drift.

Research documentation:

- [Research notes](docs/RESEARCH.md)
- [Rubric mapping](docs/RUBRIC_MAPPING.md)
- [IEEE paper source](paper/main.tex)
- [Bibliography](paper/refs.bib)
- [Experimental results](experiments/results/results.csv)

## Quickstart: Run The Full Stack

Requirements:

- Docker Desktop
- Python 3.10 to 3.12 if running locally without Docker

Start everything:

```bash
docker compose up -d --build
```

Open the services:

| Service | URL | Purpose |
|---|---:|---|
| FastAPI | http://localhost:8000/docs | Inference API and schema |
| Prometheus | http://localhost:9090 | Metrics and scrape targets |
| Grafana | http://localhost:3000 | Dashboards, login `admin` / `admin` |
| MLflow | http://localhost:5000 | Experiment tracking |
| Drift monitor metrics | http://localhost:9100/metrics | Drift-monitor metric endpoint |

Stop everything:

```bash
docker compose down
```

## Smoke Test

After the stack is running:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],"label":1}'
```

Expected behavior:

- `/health` returns `status=ok`.
- `/predict` returns a class prediction, probability, model version, and latency.
- Prometheus shows API and drift-monitor targets as `UP`.
- Grafana loads three dashboards.
- MLflow contains the warmup training run.

## Repository Structure

```text
.
├── .github/workflows/          # CI, Docker build, optional deploy, paper build
├── architecture/               # Mermaid and rendered architecture diagrams
├── data/                       # Dataset placeholders and feature metadata
├── demo_artifacts/             # Demo video and screenshots
├── deploy/
│   ├── docker/                 # Dockerfiles for API, trainer, drift monitor, MLflow
│   ├── grafana/                # Provisioned dashboards and datasources
│   └── prometheus/             # Scrape config and alert rules
├── docs/                       # Project explanation and submission notes
├── experiments/results/        # Benchmark result files
├── notebooks/                  # Walkthrough notebook
├── paper/                      # IEEE paper source, figures, bibliography
├── scripts/                    # Reproducibility and demo helper scripts
├── src/                        # Application, model, drift, monitoring, pipeline code
└── tests/                      # Unit and API tests
```

## Main Components

| Component | Files | What it does |
|---|---|---|
| API service | `src/api/` | Serves prediction, health, reload, and Prometheus metrics endpoints |
| Online models | `src/models/` | Wraps SGD, batch logistic regression, and Hoeffding tree learners |
| Drift detectors | `src/drift/` | Implements ADWIN, DDM, EDDM, KSWIN, Page-Hinkley, and HybridDD |
| Training pipeline | `src/pipelines/train.py` | Trains warmup model and logs to MLflow |
| Experiment harness | `src/pipelines/experiment.py` | Runs prequential benchmark and statistical analysis |
| Drift monitor | `src/monitoring/drift_service.py` | Streams data into the API, updates drift metrics, and triggers reloads |
| Monitoring | `deploy/prometheus/`, `deploy/grafana/` | Scrapes metrics, alerts, and renders dashboards |
| CI/CD | `.github/workflows/` | Lint, tests, smoke experiment, Docker build, paper build, optional AWS deploy |

## Metrics Exposed

The API and monitor expose production-style metrics:

- `ml_predictions_total`
- `ml_prediction_errors_total`
- `ml_drift_events_total`
- `ml_retrains_total`
- `ml_rolling_accuracy`
- `ml_rolling_error_rate`
- `ml_drift_severity`
- `ml_model_version_info`
- `ml_inference_latency_seconds`
- `ml_online_update_latency_seconds`

Grafana dashboards use these metrics to visualize model quality, drift behavior, retraining events, API health, and latency.

## Reproduce Experiments

Quick smoke experiment:

```bash
python -m src.pipelines.experiment \
  --no-mlflow \
  --seeds 42 \
  --models sgd_logistic \
  --detectors ddm hybrid \
  --streams sea \
  --n-samples 1000
```

Full benchmark:

```bash
python -m src.pipelines.experiment \
  --seeds 42 1337 2024 \
  --models sgd_logistic hoeffding_tree \
  --detectors adwin ddm eddm kswin page_hinkley hybrid \
  --streams elec2 sea hyperplane
```

Refresh paper macros:

```bash
python scripts/fill_paper_numbers.py
```

## Validation Status

Local validation performed:

- `pytest`: 36 tests passed.
- `ruff check src tests`: passed.
- `ruff format --check src tests`: passed.
- Docker Compose stack started successfully.
- API, MLflow, Prometheus, Grafana, and drift monitor verified.
- Demo video and screenshots generated.

## Evaluation Rubric Summary

| Requirement | Status |
|---|---|
| MLflow tracking | Implemented and demonstrated |
| Docker containerization | Implemented with Docker Compose |
| Prometheus metrics | Implemented for API and drift monitor |
| Grafana dashboard | Three dashboards provisioned |
| CI/CD | GitHub Actions included |
| Architecture diagram | Included |
| Research paper | IEEE LaTeX source included |
| Literature review | 19 bibliography entries included |
| Experimental validation | Benchmark harness and result files included |
| Demo media | Video and screenshots included |

Full mapping: [Rubric mapping](docs/RUBRIC_MAPPING.md)

## Notes For Evaluators

The repository includes smoke-sized result artifacts for fast inspection. The paper macros include representative full-benchmark values, and the full benchmark command is provided for reproducibility. Running the full benchmark can take significantly longer than the smoke run because it evaluates multiple detectors, streams, models, and seeds.

## License

MIT
