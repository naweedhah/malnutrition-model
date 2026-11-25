"""HTTP API for the Monitor stage: "has the data the model sees in the field drifted?"

The integrated research backend (and anyone else) can POST a reference sample (what the model
was trained or last verified on) and a current sample (recent field data) and get back, per
feature, the PSI / KS / missingness results and whether that feature is flagged.

Run locally:
    uvicorn component3.api:app --reload
    open http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, Union

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.alias_generators import to_camel

from component3.config import Thresholds
from component3.interfaces import FeatureSchema, FeatureSpec
from component3.monitor.monitor import FeatureShiftResult, run_monitoring

MIN_ROWS = 30  # below this, PSI and KS results are too noisy to act on
MAX_ROWS = 50_000  # keeps a single request's memory and CPU bounded
MAX_FEATURES = 200

Cell = Union[float, int, str, bool, None]


class CamelModel(BaseModel):
    """JSON uses camelCase; Python uses snake_case."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class FeatureIn(CamelModel):
    name: Annotated[str, Field(min_length=1, max_length=64)]
    kind: Literal["numeric", "categorical"]


class ThresholdsIn(CamelModel):
    psi_flag: Annotated[float, Field(gt=0, le=5)] = Thresholds.psi_flag
    ks_alpha: Annotated[float, Field(gt=0, lt=1)] = Thresholds.ks_alpha
    missingness_delta_flag: Annotated[float, Field(ge=0, le=1)] = Thresholds.missingness_delta_flag


class DriftRequest(CamelModel):
    features: Annotated[list[FeatureIn], Field(min_length=1, max_length=MAX_FEATURES)]
    reference: Annotated[list[dict[str, Cell]], Field(min_length=MIN_ROWS, max_length=MAX_ROWS)]
    current: Annotated[list[dict[str, Cell]], Field(min_length=MIN_ROWS, max_length=MAX_ROWS)]
    thresholds: ThresholdsIn = ThresholdsIn()

    @field_validator("features")
    @classmethod
    def unique_names(cls, features: list[FeatureIn]) -> list[FeatureIn]:
        names = [f.name for f in features]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate feature names: {duplicates}")
        return features

    @model_validator(mode="after")
    def columns_present_and_typed(self) -> "DriftRequest":
        for label, rows in (("reference", self.reference), ("current", self.current)):
            seen = set().union(*(row.keys() for row in rows))
            missing = [f.name for f in self.features if f.name not in seen]
            if missing:
                raise ValueError(f"{label} rows never contain feature(s) {missing}")
            for feature in self.features:
                if feature.kind != "numeric":
                    continue
                for i, row in enumerate(rows):
                    value = row.get(feature.name)
                    if value is not None and (isinstance(value, (str, bool)) or not math.isfinite(value)):
                        raise ValueError(
                            f"{label}[{i}].{feature.name} must be a finite number or null, got {value!r}"
                        )
        return self


class FeatureResultOut(CamelModel):
    feature: str
    kind: str
    psi: float | None
    ks_statistic: float | None
    ks_p_value: float | None
    missingness_delta: float
    flagged: bool
    reasons: list[str]


class DriftResponse(CamelModel):
    any_flagged: bool
    flagged_features: list[str]
    reference_rows: int
    current_rows: int
    features: list[FeatureResultOut]


def _finite(value: float | None) -> float | None:
    """JSON has no NaN; report undefined statistics as null."""
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def _to_out(r: FeatureShiftResult) -> FeatureResultOut:
    return FeatureResultOut(
        feature=r.feature,
        kind=r.kind,
        psi=_finite(r.psi_value),
        ks_statistic=_finite(r.ks_statistic),
        ks_p_value=_finite(r.ks_pvalue),
        missingness_delta=float(r.missingness_delta),
        flagged=r.flagged,
        reasons=list(r.reasons),
    )


app = FastAPI(
    title="Component 3: Model Reliability Monitoring",
    version="0.2.0",
    description=(
        "Detects distribution shift between the data a deployed model was validated on and the "
        "data it now receives (PSI, two-sample KS test, missingness change)."
    ),
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/drift", response_model=DriftResponse, response_model_by_alias=True)
def detect_drift(request: DriftRequest) -> DriftResponse:
    schema = FeatureSchema(tuple(FeatureSpec(name=f.name, kind=f.kind) for f in request.features))
    columns = schema.columns
    reference = pd.DataFrame(request.reference).reindex(columns=columns)
    current = pd.DataFrame(request.current).reindex(columns=columns)
    for spec in schema.features:
        if spec.kind == "numeric":
            reference[spec.name] = pd.to_numeric(reference[spec.name])
            current[spec.name] = pd.to_numeric(current[spec.name])

    thresholds = Thresholds(
        psi_flag=request.thresholds.psi_flag,
        ks_alpha=request.thresholds.ks_alpha,
        missingness_delta_flag=request.thresholds.missingness_delta_flag,
    )
    report = run_monitoring(reference, current, schema, thresholds)

    return DriftResponse(
        any_flagged=report.any_flagged,
        flagged_features=report.flagged_features(),
        reference_rows=len(reference),
        current_rows=len(current),
        features=[_to_out(r) for r in report.feature_results],
    )
