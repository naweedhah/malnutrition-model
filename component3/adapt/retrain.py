"""Retraining action (spec §11, Action 3): the more consequential maintenance option.

`model_factory` returns an unfitted sklearn-compatible estimator so C3 stays
model-family-agnostic (logistic regression, Random Forest, XGBoost, CatBoost per
spec §4.4 are all valid factories).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from component3.adapt.candidate import CandidateModel
from component3.interfaces import FeatureSchema, ModelMetadata


def generate_retrained_candidate(
    model_factory: Callable[[], object],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    next_version: str,
    parent_version: str,
    schema: FeatureSchema | None = None,
) -> CandidateModel:
    estimator = model_factory()
    estimator.fit(X_train, y_train)
    classes = getattr(estimator, "classes_", np.unique(y_train))

    metadata = ModelMetadata(
        version=next_version,
        classes=tuple(np.asarray(classes).tolist()),
        parent_version=parent_version,
        action="retrain",
        schema=schema,
    )
    return CandidateModel(estimator=estimator, version=next_version, metadata=metadata)
