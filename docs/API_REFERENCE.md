# API Reference

Base URL:

```text
http://localhost:8000
```

Interactive docs:

```text
http://localhost:8000/docs
```

## `GET /health`

Checks API readiness and model load status.

Example:

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "local",
  "uptime_s": 18.75
}
```

## `POST /predict`

Runs prediction. If `label` is supplied, the online model updates itself and error counters are updated.

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],"label":1}'
```

Request body:

```json
{
  "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
  "label": 1
}
```

Response:

```json
{
  "prediction": 1,
  "probability": 1.0,
  "model_version": "local",
  "inference_us": 401.8
}
```

## `POST /reload`

Reloads the deployed model. The drift monitor calls this endpoint after drift events.

Example:

```bash
curl -X POST http://localhost:8000/reload
```

Response:

```json
{
  "status": "reloaded",
  "new_version": "local"
}
```

## `GET /metrics`

Exposes project-specific Prometheus metrics.

```bash
curl http://localhost:8000/metrics
```

Important metrics:

- `ml_predictions_total`
- `ml_prediction_errors_total`
- `ml_model_version_info`
- `ml_inference_latency_seconds`
- `ml_online_update_latency_seconds`

## `GET /metrics-fastapi`

Exposes request-level FastAPI instrumentation from `prometheus-fastapi-instrumentator`.

```bash
curl http://localhost:8000/metrics-fastapi
```
