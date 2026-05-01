# Demo Video

This folder now contains two kinds of evidence:

1. `live_localhost_walkthrough.mp4` is the proper live demo. It records the project running from real localhost services.
2. `mlops_live_demo.mp4` is the older screenshot/evidence review video.

## Watch Or Download

- [Download / open the real localhost MP4](https://github.com/abubakarshahid16/drift-aware-mlops-pipeline/raw/main/demo_artifacts/live_localhost_walkthrough.mp4)
- [Download / open the smaller WebM](https://github.com/abubakarshahid16/drift-aware-mlops-pipeline/raw/main/demo_artifacts/live_localhost_walkthrough.webm)
- [Open the MP4 file page](live_localhost_walkthrough.mp4)
- [View screenshots](screenshots/)

GitHub may show a message like this on large video file pages:

```text
Sorry about that, but we can't show files that are this big right now.
```

That does **not** mean the video is missing. Use the raw/download links above.

## Preview

![Live localhost demo preview](live_localhost_preview.gif)

## What The Proper Localhost Video Shows

1. FastAPI Swagger docs at `http://localhost:8000/docs`.
2. API health response at `http://localhost:8000/health`.
3. Live `POST /predict` response from the running API.
4. Prometheus scrape targets at `http://localhost:9090/targets`.
5. Live Prometheus query for `ml_predictions_total`.
6. MLflow tracking evidence from `http://localhost:5000`.
7. Grafana model performance dashboard at `http://localhost:3000`.
8. Grafana drift and adaptive retraining dashboard.
9. Grafana infrastructure and API health dashboard.

## Video Metadata

- File: `live_localhost_walkthrough.mp4`
- Browser source: real local services on ports `8000`, `5000`, `9090`, `3000`, and `9100`
- Duration: about 60 seconds
- Resolution: 1366x768
- MP4 size: about 13.2 MB
- WebM size: about 3.2 MB
