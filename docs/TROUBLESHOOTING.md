# Troubleshooting

## Docker Desktop Is Not Running

Symptom:

```text
Cannot connect to the Docker daemon
```

Fix:

```bash
docker desktop start
```

Then:

```bash
docker compose up -d --build
```

## Port Already In Use

Ports used:

- 8000 API
- 5000 MLflow
- 9090 Prometheus
- 3000 Grafana
- 9100 drift monitor

Find the process using a port on Windows:

```powershell
netstat -ano | findstr :8000
```

Stop the old service or change the port in `docker-compose.yml`.

## Grafana Login

Default credentials:

```text
admin / admin
```

If Grafana asks to change password, choose skip for local demo use.

## MLflow Loads Slowly In Local Development

The API first tries MLflow. If MLflow is not reachable, the model loader falls back to local artifacts. The code includes a reachability check to avoid long MLflow client retries.

Relevant file:

```text
src/api/model_registry.py
```

## Drift Monitor Target Shows Down

The Docker drift monitor uses `--hold`, so it should keep `/metrics` alive after stream replay. If the target is down:

```bash
docker compose logs drift_monitor --tail=100
docker compose up -d --build drift_monitor
```

Then open:

```text
http://localhost:9090/targets
```

## Python 3.13 Dependency Problems

Use Python 3.10 to 3.12. The project pins libraries that are tested on those versions.

Check Python:

```bash
python --version
```

## Dataset Missing

Run:

```bash
python -m src.data.download
python -m src.data.preprocess
```

Docker runs these steps automatically through the trainer service.

## Demo Video Does Not Preview In Browser

GitHub may sometimes show a large video as a downloadable file rather than inline preview. The proper localhost walkthrough MP4 is still included at:

```text
demo_artifacts/live_localhost_walkthrough.mp4
```

A smaller WebM version is also included at:

```text
demo_artifacts/live_localhost_walkthrough.webm
```

Screenshots are also included in:

```text
demo_artifacts/screenshots/
```
