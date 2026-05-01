# Project Quality Checklist

Use this checklist before final submission.

## Technical

- [x] Docker Compose runs the full stack.
- [x] API has health and prediction endpoints.
- [x] MLflow tracks training metrics and artifacts.
- [x] Prometheus scrapes API and drift monitor metrics.
- [x] Grafana dashboards are provisioned automatically.
- [x] Drift monitor exposes model-quality and drift metrics.
- [x] CI workflow runs lint, formatting, tests, and smoke experiment.
- [x] Docker workflow builds all service images.
- [x] Optional deployment workflow is included.

## Research

- [x] Research problem is clearly stated.
- [x] HybridDD contribution is explained.
- [x] Literature review references are included.
- [x] Evaluation metrics are defined.
- [x] Experiment harness is implemented.
- [x] Statistical test code is included.
- [x] IEEE paper source is included.

## Documentation

- [x] README explains what the project is.
- [x] Setup guide is included.
- [x] Demo guide is included.
- [x] Architecture guide is included.
- [x] Monitoring guide is included.
- [x] API reference is included.
- [x] Troubleshooting guide is included.
- [x] Rubric mapping is included.
- [x] Screenshots are included.
- [x] Demo video is included.

## Final Optional Polish

- [ ] Run full benchmark and update `experiments/results/*`.
- [x] Compile `paper/main.tex` into final PDF.
- [ ] Add a GitHub release with the demo video and final paper PDF.
