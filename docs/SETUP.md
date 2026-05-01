# Setup Guide

This guide explains how to run the project from a fresh clone.

## 1. Clone The Repository

```bash
git clone https://github.com/abubakarshahid16/drift-aware-mlops-pipeline.git
cd drift-aware-mlops-pipeline
```

## 2. Run With Docker

Docker is the recommended path because it starts all services with one command.

```bash
docker compose up -d --build
```

Wait until the trainer finishes and the API becomes healthy:

```bash
docker compose ps
```

Expected services:

- `postgres`
- `mlflow`
- `trainer`
- `api`
- `drift_monitor`
- `prometheus`
- `grafana`

The trainer is a one-shot job, so it may show `Exited (0)`. That is expected.

## 3. Open The Services

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000/docs |
| API health | http://localhost:8000/health |
| Prometheus | http://localhost:9090 |
| Prometheus targets | http://localhost:9090/targets |
| Grafana | http://localhost:3000 |
| MLflow | http://localhost:5000 |
| Drift monitor metrics | http://localhost:9100/metrics |

Grafana credentials:

```text
admin / admin
```

## 4. Smoke Test

```bash
curl http://localhost:8000/health
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],"label":1}'
```

## 5. Run Locally Without Docker

Use Python 3.10 to 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Prepare data and train:

```bash
python -m src.data.download
python -m src.data.preprocess
python -m src.pipelines.train --no-register
```

Run API:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Run drift monitor in a second terminal:

```bash
python -m src.monitoring.drift_service --api-url http://localhost:8000 --metrics-port 9100 --hold
```

## 6. Run Tests

```bash
python -m ruff check src tests
python -m ruff format --check src tests
python -m pytest -q
```

## 7. Stop Docker Stack

```bash
docker compose down
```

To remove volumes as well:

```bash
docker compose down -v
```
