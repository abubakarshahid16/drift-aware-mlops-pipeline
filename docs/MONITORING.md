# Monitoring And Observability

## Monitoring Goals

The monitoring layer is not decorative. It is part of the research design because drift signals are exposed as operational metrics and used to drive adaptation.

## Prometheus Metrics

| Metric | Type | Meaning |
|---|---|---|
| `ml_predictions_total` | Counter | Number of predictions by predicted label |
| `ml_prediction_errors_total` | Counter | Number of labeled prediction errors |
| `ml_drift_events_total` | Counter | Drift events by detector and kind |
| `ml_retrains_total` | Counter | Reload/retrain triggers by reason |
| `ml_rolling_accuracy` | Gauge | Rolling accuracy from recent labeled examples |
| `ml_rolling_error_rate` | Gauge | Rolling error rate from recent labeled examples |
| `ml_drift_severity` | Gauge | Last drift severity |
| `ml_model_version_info` | Gauge | Deployed model version/source label |
| `ml_inference_latency_seconds` | Histogram | Prediction latency |
| `ml_online_update_latency_seconds` | Histogram | Online update latency |

## Grafana Dashboards

The repository provisions three dashboards automatically:

1. Model Performance
   - Rolling accuracy
   - Rolling error rate
   - Predictions by class
   - Inference latency percentiles

2. Drift And Adaptive Retraining
   - Drift events
   - Drift severity
   - Retraining triggers
   - Rolling quality changes

3. Infrastructure And API Health
   - Prometheus target health
   - Request throughput
   - Request latency
   - API status metrics

## Alerting

Prometheus alert rules are defined in `deploy/prometheus/alerts.yml`. They cover sustained low accuracy and high drift activity.

## Demo Evidence

Screenshots are stored in `demo_artifacts/screenshots/`.

The most useful monitoring screenshots are:

- `05_prometheus_targets.png`
- `06_prometheus_prediction_metrics.png`
- `08_grafana_model_performance.png`
- `09_grafana_drift_dashboard.png`
- `10_grafana_infra_dashboard.png`
