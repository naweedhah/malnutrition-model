"""HTTP contract of the drift API (component3.api)."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from component3.api import MIN_ROWS, app

client = TestClient(app)

FEATURES = [
    {"name": "weightForAgeZ", "kind": "numeric"},
    {"name": "district", "kind": "categorical"},
]


def rows(n: int, *, z_mean: float = 0.0, districts=("Colombo", "Kandy"), missing_every: int = 0, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        z = None if missing_every and i % missing_every == 0 else float(rng.normal(z_mean, 1.0))
        out.append({"weightForAgeZ": z, "district": districts[i % len(districts)]})
    return out


def post(body):
    return client.post("/v1/drift", json=body)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_same_distribution_is_not_flagged():
    res = post({"features": FEATURES, "reference": rows(400, seed=1), "current": rows(400, seed=2)})

    assert res.status_code == 200
    body = res.json()
    assert body["anyFlagged"] is False
    assert body["flaggedFeatures"] == []
    assert {f["feature"] for f in body["features"]} == {"weightForAgeZ", "district"}


def test_shifted_numeric_feature_is_flagged_with_reasons():
    res = post({"features": FEATURES, "reference": rows(400, seed=1), "current": rows(400, z_mean=-1.5, seed=2)})

    body = res.json()
    z = next(f for f in body["features"] if f["feature"] == "weightForAgeZ")
    assert body["anyFlagged"] is True
    assert "weightForAgeZ" in body["flaggedFeatures"]
    assert z["psi"] > 0.2 and z["ksPValue"] < 0.05
    assert any(reason.startswith("psi=") for reason in z["reasons"])


def test_new_category_mix_and_missingness_are_flagged():
    res = post({
        "features": FEATURES,
        "reference": rows(400, seed=1),
        "current": rows(400, districts=("Jaffna", "Jaffna", "Kandy"), missing_every=4, seed=2),
    })

    flagged = set(res.json()["flaggedFeatures"])
    assert flagged == {"weightForAgeZ", "district"}


def test_thresholds_can_be_tuned_per_request():
    body = {"features": FEATURES, "reference": rows(400, seed=1), "current": rows(400, z_mean=-1.5, seed=2),
            "thresholds": {"psiFlag": 5, "ksAlpha": 1e-300, "missingnessDeltaFlag": 1}}

    assert post(body).json()["anyFlagged"] is False


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda b: b.update(reference=rows(MIN_ROWS - 1)), "at least 30"),
        (lambda b: b.update(features=FEATURES + [FEATURES[0]]), "duplicate feature names"),
        (lambda b: b.update(features=[{"name": "heightCm", "kind": "numeric"}]), "never contain feature"),
        (lambda b: b["current"][3].update(weightForAgeZ="low"), "finite number or null"),
        (lambda b: b.update(features=[{"name": "x", "kind": "ordinal"}]), "numeric"),
        (lambda b: b.update(thresholds={"ksAlpha": 2}), "less than 1"),
        (lambda b: b.update(unexpected=True), "Extra inputs are not permitted"),
    ],
)
def test_invalid_requests_get_422_with_a_reason(mutate, message):
    body = {"features": FEATURES, "reference": rows(MIN_ROWS), "current": rows(MIN_ROWS)}
    mutate(body)

    res = post(body)

    assert res.status_code == 422
    assert message in str(res.json()["detail"])
