# Adaptive Model Reliability Monitoring and Safe Maintenance (Component 3)

[![CI](https://github.com/naweedhah/malnutrition-model/actions/workflows/ci.yml/badge.svg)](https://github.com/naweedhah/malnutrition-model/actions/workflows/ci.yml)

My component of a final-year research project on AI decision support for **childhood
malnutrition in Sri Lanka**. Component 1 predicts a child's malnutrition risk. This component
answers the next question: **can the deployed model still be trusted, and if not, how do we
update it without making things worse?**

Field data changes over time: new districts, different measurement practice, seasonal food
insecurity. A model validated last year can silently become unreliable. Retraining whenever
something looks different is also risky. This framework closes the loop safely:

```
 Monitor ──► Assess ──► Decide ──► Adapt ──► Verify ──► Promote / Reject
 data shift  is the     no action, recalibrate  candidate   registry keeps
 (PSI, KS,   model      recalibrate, or retrain  must pass   every version
 missingness) still     retrain or a  a candidate fixed gates and the reason
             reliable?  hold         model       on held-out  it was promoted
                                                  data         or rejected
```

| Stage | Module | What it does |
| --- | --- | --- |
| Monitor | `component3/monitor` | Per-feature **Population Stability Index**, two-sample **Kolmogorov–Smirnov** test, missing-value drift, and prediction-distribution PSI |
| Assess | `component3/assess` | Macro recall, minority-class recall, **Expected Calibration Error**, multi-class **Brier score** on newly labelled outcomes |
| Decide | `component3/decide` | Conservative policy: `NO_ACTION`, `RECALIBRATE` (ranking still fine, probabilities off), `RETRAIN` (only with enough labels), or `HOLD` (shift seen but no labels yet) |
| Adapt | `component3/adapt` | Builds a recalibrated or retrained **candidate**; never touches the live model |
| Verify | `component3/verify` | Candidate must meet absolute floors **and** not regress more than an allowed margin against the active model on the same untouched split |
| Registry | `component3/registry` | Versioned models with a full promotion and rejection history (audit trail) |
| Experiments | `component3/experiments` | Stress scenarios (covariate shift, label-prevalence shift, missing values, measurement noise, out-of-range values, a feature becoming unavailable) comparing *never update*, *always recalibrate*, *always retrain* and this adaptive policy |

The framework only depends on the model contract in [`interfaces.py`](component3/interfaces.py)
(`predict` / `predict_proba` plus a feature schema), so it can maintain any classifier.

## Drift-monitoring API

The Monitor stage is also exposed over HTTP with **FastAPI**, so the project's integrated
backend (or any service) can ask "has this data drifted?":

```bash
pip install -e ".[api]"
uvicorn component3.api:app --reload      # interactive docs at http://127.0.0.1:8000/docs
# or
docker build -t component3-api . && docker run -p 8000:8000 component3-api
```

```http
POST /v1/drift
{
  "features":  [{"name": "weightForAgeZ", "kind": "numeric"}, {"name": "district", "kind": "categorical"}],
  "reference": [{"weightForAgeZ": -0.4, "district": "Colombo"}, ...],   // at least 30 rows
  "current":   [{"weightForAgeZ": -1.9, "district": "Jaffna"}, ...],
  "thresholds": {"psiFlag": 0.2, "ksAlpha": 0.05, "missingnessDeltaFlag": 0.1}   // optional
}
→ 200 { "anyFlagged": true, "flaggedFeatures": ["weightForAgeZ"], "features": [{ "psi": 1.42, "ksPValue": 1e-40, "reasons": ["psi=1.420>=0.2", ...] }, ...] }
```

Requests are validated with Pydantic: unique feature names, every feature present in both
samples, finite numbers for numeric features, 30–50,000 rows, and sane threshold ranges. Anything
else gets a **422** with the exact reason.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

53 tests cover the detectors (PSI and KS behaviour on known shifts), metrics (ECE, Brier),
each decision branch, recalibration and retraining, the verification gates, registry history,
the full orchestrated cycle, the experiment harness, and the HTTP contract. CI runs them on
Python 3.11 and 3.13 and checks that the Docker image starts and answers `/health`.

## Data

No real child-health data is stored in this repository. The tests use synthetic data from
scikit-learn's `make_classification`.
