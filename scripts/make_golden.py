"""
Regenerates test/fixtures/sklearn-golden.json - the reference results the TypeScript solvers are
tested against. Only needed when adding cases or re-checking against a new scikit-learn release;
the fixture is checked in, so normal development and CI need no Python.

Reference solver is scikit-learn's newton-cholesky (exact Newton, like ours) at a tight tolerance,
so any difference measures our implementation, not an under-converged reference.

  python3 -m venv .venv && .venv/bin/pip install numpy scikit-learn
  .venv/bin/python scripts/make_golden.py
"""

import json
import os

import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

OUT = os.path.join(os.path.dirname(__file__), "..", "test", "fixtures", "sklearn-golden.json")


def reference_lr(**kwargs):
    return LogisticRegression(solver="newton-cholesky", tol=1e-12, max_iter=1000, **kwargs)


def lr_case(name, X, y, *, C, class_weight=None, sample_weight=None):
    model = reference_lr(C=C, class_weight=class_weight).fit(X, y, sample_weight=sample_weight)
    return {
        "name": name, "C": C, "class_weight": class_weight,
        "X": X.tolist(), "y": y.tolist(),
        "sample_weight": None if sample_weight is None else sample_weight.tolist(),
        "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
        "proba": model.predict_proba(X)[:, 1].tolist(),
    }


def main():
    rng = np.random.default_rng(0)
    X, y = make_classification(n_samples=200, n_features=12, n_informative=6, weights=[0.85],
                               flip_y=0.05, random_state=0)
    sw = rng.uniform(0.1, 3.0, size=len(y))

    cases = [
        lr_case("balanced_C1", X, y, C=1.0, class_weight="balanced"),
        lr_case("sample_weight_C0.3", X, y, C=0.3, sample_weight=sw),
        lr_case("strong_reg_C0.01_balanced", X, y, C=0.01, class_weight="balanced", sample_weight=sw),
    ]

    # Platt exactly as train.py: weighted LR (C=1e6) on the classifier's logit, prevalence weights.
    base = reference_lr(C=1.0, class_weight="balanced").fit(X, y)
    logits = base.decision_function(X)
    prevalence = 0.01
    observed = y.mean()
    pw = np.where(y == 1, prevalence / observed, (1 - prevalence) / (1 - observed))
    platt = reference_lr(C=1e6).fit(logits.reshape(-1, 1), y, sample_weight=pw)

    # Isotonic on the raw probability, rounded so there are many duplicate x values to merge.
    raw = np.round(1 / (1 + np.exp(-logits)), 2)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y, sample_weight=pw)
    grid = np.linspace(-0.1, 1.1, 61)

    # Same isotonic problem with some zero weights: scikit-learn drops those samples before fitting.
    zw = pw.copy()
    zw[::7] = 0.0
    iso_zw = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y, sample_weight=zw)

    fixture = {
        "generated_with": {"scikit_learn": sklearn.__version__, "numpy": np.__version__},
        "logistic": cases,
        "platt": {"logits": logits.tolist(), "y": y.tolist(), "sample_weight": pw.tolist(),
                  "a": float(platt.coef_[0][0]), "c": float(platt.intercept_[0])},
        "isotonic": {"x": raw.tolist(), "y": y.tolist(), "sample_weight": pw.tolist(),
                     "x_thresholds": iso.X_thresholds_.tolist(), "y_thresholds": iso.y_thresholds_.tolist(),
                     "grid": grid.tolist(), "grid_predictions": iso.predict(grid).tolist()},
        "isotonic_zero_weight": {"x": raw.tolist(), "y": y.tolist(), "sample_weight": zw.tolist(),
                                 "x_thresholds": iso_zw.X_thresholds_.tolist(),
                                 "y_thresholds": iso_zw.y_thresholds_.tolist(),
                                 "grid": grid.tolist(), "grid_predictions": iso_zw.predict(grid).tolist()},
    }
    with open(OUT, "w") as f:
        json.dump(fixture, f)
    print(f"wrote {OUT} ({os.path.getsize(OUT) // 1024} KB) with scikit-learn {sklearn.__version__}")


if __name__ == "__main__":
    main()
