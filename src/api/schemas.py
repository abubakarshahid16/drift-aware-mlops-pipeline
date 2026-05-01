from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class PredictRequest(ApiModel):
    features: list[float] = Field(..., min_length=1, max_length=64)
    label: int | None = Field(
        None, ge=0, le=1, description="Optional ground-truth for online update."
    )
    request_id: str | None = None


class PredictResponse(ApiModel):
    prediction: int
    probability: float
    model_version: str
    inference_us: float


class HealthResponse(ApiModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
    uptime_s: float


class ReloadResponse(ApiModel):
    status: Literal["reloaded", "failed"]
    new_version: str
    detail: str | None = None
