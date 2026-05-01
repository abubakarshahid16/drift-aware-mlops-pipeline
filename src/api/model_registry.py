"""Thin loader that prefers MLflow Registry, falls back to a local joblib artifact.

Keeping this isolated lets the API run in CI/dev without MLflow available, and
keeps the registry coupling at a single seam.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import joblib

from src.models.base import OnlineModel
from src.utils.config import EXPERIMENTS_DIR, settings
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class LoadedModel:
    model: OnlineModel
    version: str
    source: str


def _local_fallback() -> LoadedModel | None:
    local = EXPERIMENTS_DIR / "models" / "sgd_logistic.joblib"
    if local.exists():
        model = joblib.load(local)
        log.info("Loaded local model from %s", local)
        return LoadedModel(model=model, version="local", source="local")
    return None


def _tracking_server_reachable(uri: str, timeout_s: float = 0.35) -> bool:
    parsed = urlparse(uri)
    if parsed.scheme not in {"http", "https"}:
        return True

    host = parsed.hostname
    if host is None:
        return False

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def load() -> LoadedModel:
    # Try MLflow first. Use search_model_versions to avoid the deprecated
    # stage-based registry calls (MLflow >= 2.9).
    try:
        if not _tracking_server_reachable(settings.mlflow_tracking_uri):
            log.warning(
                "MLflow tracking server %s is not reachable, falling back to local artifact",
                settings.mlflow_tracking_uri,
            )
            fallback = _local_fallback()
            if fallback is not None:
                return fallback

        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{settings.model_name}'", max_results=10)
        if versions:
            mv = max(versions, key=lambda v: int(v.version))
            local_path = mlflow.artifacts.download_artifacts(
                f"models:/{settings.model_name}/{mv.version}"
            )
            joblib_path = next(Path(local_path).rglob("*.joblib"), None)
            if joblib_path is not None:
                model: OnlineModel = joblib.load(joblib_path)
                log.info("Loaded model %s v%s from MLflow", settings.model_name, mv.version)
                return LoadedModel(model=model, version=str(mv.version), source="mlflow")
    except Exception as e:
        log.warning("MLflow load failed (%s), falling back to local artifact", e)

    # Local fallback.
    fallback = _local_fallback()
    if fallback is not None:
        return fallback

    raise RuntimeError("No model available. Train one with `python -m src.pipelines.train` first.")
