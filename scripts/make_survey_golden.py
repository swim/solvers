"""
Regenerates test/fixtures/survey-golden.json - reference results for src/survey.ts from samplics'
TaylorEstimator (stratified simple random sampling without replacement, with finite-population
correction). Development only; the fixture is checked in.

samplics' `fpc` argument is the multiplier (1 - n_h / N_h), not the sampling fraction.

  python3 -m venv .venv && .venv/bin/pip install numpy samplics
  .venv/bin/python scripts/make_survey_golden.py
"""

import importlib.metadata as md
import json
import os
import warnings

import numpy as np

warnings.simplefilter("ignore", FutureWarning)  # samplics is archived; the estimators are unchanged
from samplics.estimation import TaylorEstimator  # noqa: E402
from samplics.utils.types import PopParam  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "test", "fixtures", "survey-golden.json")


def estimate(param, y, w, strata, fpc, x=None):
    est = TaylorEstimator(param=param)
    est.estimate(y=y, x=x, samp_weight=w, stratum=strata, fpc=fpc)
    return float(est.point_est), float(est.stderror)


def case(name, rng, sizes, n, prevalence, census=()):
    strata, pis, scores, y = [], [], [], []
    for h, (N, nh) in enumerate(zip(sizes, n)):
        nh = N if h in census else nh
        strata += [f"s{h}"] * nh
        pis += [nh / N] * nh
        p = prevalence[h]
        yy = (rng.random(nh) < p).astype(int)
        y += yy.tolist()
        scores += np.where(yy == 1, rng.beta(5, 2, nh), rng.beta(2, 5, nh)).tolist()
    strata, pis, scores, y = np.array(strata), np.array(pis), np.array(scores), np.array(y, dtype=float)
    w = 1 / pis
    stratum_sizes = {f"s{h}": int(N) for h, N in enumerate(sizes)}
    fpc = {s: 1 - (strata == s).sum() / stratum_sizes[s] for s in stratum_sizes}
    total, total_se = estimate(PopParam.total, y, w, strata, fpc)
    # A miss-rate domain ratio at a threshold: positives scoring below t over positives.
    t = float(np.quantile(scores[y == 1], 0.2)) if (y == 1).any() else 0.5
    miss = y * (scores < t)
    ratio, ratio_se = estimate(PopParam.ratio, miss, w, strata, fpc, x=y)
    fired = (scores >= t).astype(float)
    precision, precision_se = estimate(PopParam.ratio, y * fired, w, strata, fpc, x=fired)
    return {
        "name": name, "strata": strata.tolist(), "inclusion_probs": pis.tolist(), "stratum_sizes": stratum_sizes,
        "y": y.astype(int).tolist(), "scores": scores.tolist(), "t": t,
        "total": total, "total_se": total_se,
        "miss_rate": ratio, "miss_rate_se": ratio_se,
        "precision": precision, "precision_se": precision_se,
    }


def main():
    rng = np.random.default_rng(20261003)
    cases = [
        case("two strata, small fractions", rng, [5000, 800], [60, 40], [0.02, 0.4]),
        case("four strata, mixed fractions", rng, [20000, 3000, 400, 90], [80, 60, 50, 45], [0.005, 0.05, 0.3, 0.8]),
        case("one census stratum", rng, [10000, 35], [100, 35], [0.01, 0.6], census=(1,)),
        case("large fractions", rng, [120, 80, 60], [90, 60, 50], [0.1, 0.3, 0.6]),
    ]
    out = {"generator": "scripts/make_survey_golden.py", "versions": {"samplics": md.version("samplics"), "numpy": np.__version__}, "cases": cases}
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
