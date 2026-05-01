# CI/CD Guide

## GitHub Actions Workflows

The repository includes four workflows.

## `ci.yml`

Runs on push and pull request.

Steps:

- Checkout repository.
- Set up Python.
- Install dev dependencies.
- Run `ruff check`.
- Run `ruff format --check`.
- Run `mypy` as a non-blocking type check.
- Run pytest on Python 3.10, 3.11, and 3.12.
- Run a minimal experiment smoke test.
- Upload coverage and smoke results as artifacts.

## `docker.yml`

Runs on push to main, tags, and manual dispatch.

Steps:

- Build API image.
- Build trainer image.
- Build drift-monitor image.
- Build MLflow image.
- Push images to GitHub Container Registry.
- Run Trivy scan.

## `deploy.yml`

Manual optional deployment workflow.

Steps:

- Authenticate to AWS with OIDC.
- Pull GHCR images.
- Retag and push to ECR.
- Force ECS service redeploy.
- Run API health smoke check.

## `paper.yml`

Runs on paper-related changes.

Steps:

- Compile IEEE LaTeX paper.
- Upload PDF artifact.

## Why This Meets CI/CD Requirements

The project is not only containerized; the CI/CD system validates code quality, tests, experiment execution, Docker image builds, security scanning, and optional cloud deployment.
