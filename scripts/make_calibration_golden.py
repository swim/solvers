"""
Regenerates test/fixtures/calibration-golden.json - reference results for src/calibration.ts:
Cox's recalibration fit from scikit-learn's unpenalised LogisticRegression (newton-cholesky) and
the likelihood-ratio test computed from it. Weights are rescaled to their Kish effective size, as
calibration.ts does. Development only.

  python3 -m venv .venv && .venv/bin/pip install numpy scipy scikit-learn
  .venv/bin/python scripts/make_calibration_golden.py
"""

import json
import os

import numpy as np
import sklearn
from scipy import stats
from sklearn.linear_model import LogisticRegression

OUT = os.path.join(os.path.dirname(__file__), "..", "test", "fixtures", "calibration-golden.json")


def kish(w):
    return w * (w.sum() ** 2 / (w ** 2).sum()) / w.sum()


def case(name, p, y, w):
    wt = kish(w)
    z = np.log(p / (1 - p))
    m = LogisticRegression(penalty=None, solver="newton-cholesky", tol=1e-14, max_iter=1000).fit(z.reshape(-1, 1), y, sample_weight=wt)
    a, b = float(m.intercept_[0]), float(m.coef_[0, 0])

    def ll(a, b):
        eta = a + b * z
        return float(np.sum(wt * (y * eta - np.logaddexp(0, eta))))

    lr = max(0.0, 2 * (ll(a, b) - ll(0, 1)))
    return {"name": name, "p": p.tolist(), "y": y.astype(int).tolist(), "w": w.tolist(), "intercept": a, "slope": b, "lr": lr,
            "p_value": float(stats.chi2.sf(lr, 2))}


def main():
    rng = np.random.default_rng(20261003)
    cases = []
    for name, n, shift, slope, weighted in [("calibrated", 2000, 0, 1, False), ("in the large", 2000, 0.5, 1, False), ("overconfident", 3000, 0, 1.4, False), ("design weights", 2500, 0.3, 1.2, True)]:
        q = 1 / (1 + np.exp(-(-3 + 1.5 * rng.standard_normal(n))))
        y = (rng.random(n) < q).astype(float)
        p = 1 / (1 + np.exp(-(slope * (np.log(q / (1 - q)) + 3) - 3 + shift)))
        w = np.where(rng.random(n) < 0.5, 4.0, 1.0) if weighted else np.ones(n)
        cases.append(case(name, p, y, w))
    with open(OUT, "w") as f:
        json.dump({"generator": "scripts/make_calibration_golden.py", "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__}, "cases": cases}, f)
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
