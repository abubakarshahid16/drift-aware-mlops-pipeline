# Rubric Mapping

## Mandatory Technical Requirements

| Requirement | Evidence |
|---|---|
| MLflow experiment tracking | `deploy/docker/Dockerfile.mlflow`, `src/pipelines/train.py`, MLflow screenshot |
| Docker containerization | `docker-compose.yml`, `deploy/docker/` |
| Prometheus metrics | `src/api/metrics.py`, `deploy/prometheus/prometheus.yml` |
| Grafana dashboard | `deploy/grafana/dashboards/` |
| CI/CD | `.github/workflows/ci.yml`, `.github/workflows/docker.yml`, `.github/workflows/deploy.yml` |
| Deployment | Local Docker deployment; optional AWS ECS workflow included |

## Research Component

| Requirement | Evidence |
|---|---|
| Research problem | `paper/main.tex`, `docs/RESEARCH.md` |
| Literature review | `paper/refs.bib` contains 19 references |
| Research questions | `docs/RESEARCH.md` |
| Evaluation metrics | `src/pipelines/metrics.py`, `paper/main.tex` |
| Experimental validation | `src/pipelines/experiment.py`, `experiments/results/` |
| Statistical analysis | Friedman and Nemenyi logic in `src/pipelines/experiment.py` |

## Deliverables

| Deliverable | Evidence |
|---|---|
| IEEE paper | `paper/main.tex` |
| GitHub repository | This repository |
| Working CI/CD pipeline | `.github/workflows/` |
| Dockerized application | `docker-compose.yml`, `deploy/docker/` |
| MLflow tracking | `mlflow` service and training pipeline |
| Monitoring dashboard | Grafana JSON dashboards |
| Architecture diagram | `architecture/architecture.png` |
| Experimental results | `experiments/results/results.csv` and `results.json` |
| Demo video | `demo_artifacts/mlops_live_demo.mp4` |
| Screenshots | `demo_artifacts/screenshots/` |

## Evaluation Criteria

| Criterion | Weight | Where to look |
|---|---:|---|
| Research novelty | 20% | HybridDD in `src/drift/detectors.py`, paper Section IV |
| Technical implementation | 25% | Docker stack, FastAPI, MLflow, monitoring services |
| Design quality | 15% | Architecture diagram, modular `src/` packages |
| Monitoring depth | 10% | Prometheus metrics, Grafana dashboards, alert rules |
| Experimental rigor | 15% | Benchmark harness, statistical tests, result artifacts |
| Documentation and presentation | 15% | README, docs, paper, screenshots, video |

## Final Status

The repository satisfies the mandatory technical stack and provides evidence for the research component. For a final graded submission, run the full benchmark if time allows and commit the full-result `stats.json` so statistical outputs are directly reproducible from the checked-in result files.
