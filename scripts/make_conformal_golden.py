"""
Regenerates test/fixtures/conformal-golden.json - the reference results conformal.ts is tested
against (src/conformal.ts). Only needed when adding cases or re-checking against new MAPIE or crepes releases; the
fixture is checked in, so normal development and CI need no Python.

  MAPIE  BinaryClassificationController with fixed-sequence testing (Learn Then Test): the PAC
         recall threshold and false-alarm threshold (risk "fpr"), plus its p-values, over every
         threshold our order statistics could pick. Recall is controlled as the MISS RATE
         (misses / positives <= alpha): MAPIE's built-in "recall" computes 1 - recall in floating
         point and then ceil(n * r), which can count one miss too many (150 * (1 - 149/150) =
         1.0000000000000064) and land a rank too conservative. Its result is recorded as
         recall.threshold_builtin for comparison.
  crepes ConformalClassifier, class-conditional (Mondrian on the label), non-smoothed p-values:
         the review floor / expected-guarantee threshold is where the positive class's p-value
         first exceeds epsilon.

  python3 -m venv .venv && .venv/bin/pip install numpy mapie crepes
  .venv/bin/python scripts/make_conformal_golden.py
"""

import importlib.metadata as md
import json
import os
import warnings

import numpy as np
from crepes import ConformalClassifier
from mapie.risk_control import BinaryClassificationController, BinaryRisk

OUT = os.path.join(os.path.dirname(__file__), "..", "test", "fixtures", "conformal-golden.json")


def proba(X):
    s = np.asarray(X, dtype=float).reshape(-1)
    return np.column_stack([1 - s, s])


# The same guarantee as recall >= 1 - alpha, stated as a risk so MAPIE sees misses / n directly.
miss_rate = BinaryRisk(
    risk_occurrence=lambda y_true, y_pred: y_pred.ravel() == 0,
    risk_condition=lambda y_true, y_pred: y_true.ravel() == 1,
    higher_is_better=False,
)


def mapie_threshold(scores, y, risk, target, delta, params):
    controller = BinaryClassificationController(
        predict_function=proba, risk=risk, target_level=target,
        confidence_level=1 - delta, list_predict_params=params, fwer_method="fixed_sequence",
        best_predict_param_choice="fpr",  # only the valid set is used; a custom risk needs an explicit choice
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "no valid threshold" is an expected outcome here
        controller.calibrate(scores.reshape(-1, 1), y)
    valid = controller.valid_predict_params
    return valid, controller.p_values[:, 0]


def case(name, pos, neg, alpha, beta, delta, epsilon, rng):
    scores = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos), int), np.zeros(len(neg), int)])

    # Recall: candidates are the distinct positive scores, ascending (miss rate rises with the threshold).
    recall_params = np.unique(pos)
    valid, recall_p = mapie_threshold(scores, y, miss_rate, alpha, delta, recall_params)
    recall_threshold = float(valid.max()) if len(valid) else None
    builtin, _ = mapie_threshold(scores, y, "recall", 1 - alpha, delta, recall_params)
    builtin_threshold = float(builtin.max()) if len(builtin) else None
    misses = [int(np.sum(pos < t)) for t in recall_params]

    # False alarms: candidates sit just above each distinct negative score (fire at p >= t).
    fa_params = np.nextafter(np.unique(neg), np.inf)
    valid, fa_p = mapie_threshold(scores, y, "fpr", beta, delta, fa_params)
    fa_threshold = float(valid.min()) if len(valid) else None
    alarms = [int(np.sum(neg >= t)) for t in fa_params]

    # crepes, class-conditional: alpha = 1 - p(label); positive-class p-values on a grid of scores.
    cc = ConformalClassifier().fit(np.where(y == 1, 1 - scores, scores), bins=y)
    grid = np.unique(np.concatenate([pos, rng.uniform(0, 1, 50)]))
    p_pos = cc.predict_p(1 - grid, bins=np.ones(len(grid), int), smoothing=False)
    p_neg = cc.predict_p(grid, bins=np.zeros(len(grid), int), smoothing=False)
    return {
        "name": name, "alpha": alpha, "beta": beta, "delta": delta, "epsilon": epsilon,
        "positives": pos.tolist(), "negatives": neg.tolist(),
        "recall": {"params": recall_params.tolist(), "misses": misses, "p_values": recall_p.tolist(), "threshold": recall_threshold, "threshold_builtin": builtin_threshold},
        "false_alarm": {"params": fa_params.tolist(), "alarms": alarms, "p_values": fa_p.tolist(), "threshold": fa_threshold},
        "crepes": {"grid": grid.tolist(), "p_positive": p_pos.tolist(), "p_negative": p_neg.tolist()},
    }


def main():
    rng = np.random.default_rng(20261003)
    cases = []
    for n_pos, n_neg, alpha, beta, delta, epsilon in [
        (30, 300, 0.10, 0.05, 0.05, 0.05),
        (59, 500, 0.05, 0.02, 0.05, 0.02),
        (150, 1000, 0.05, 0.01, 0.05, 0.03),
        (150, 1000, 0.05, 0.01, 0.10, 0.05),
        (500, 2000, 0.05, 0.005, 0.025, 0.01),
        (1000, 3000, 0.05, 0.002, 0.05, 0.02),
        (40, 400, 0.05, 0.01, 0.05, 0.05),  # too few positives for PAC at 95%
    ]:
        pos = rng.beta(5, 2, n_pos)
        neg = rng.beta(2, 5, n_neg)
        cases.append(case(f"beta n+={n_pos} n-={n_neg} a={alpha} d={delta}", pos, neg, alpha, beta, delta, epsilon, rng))
    # Ties: scores on a coarse grid, as isotonic plateaus produce.
    pos = np.round(rng.beta(5, 2, 200), 1)
    neg = np.round(rng.beta(2, 5, 800), 1)
    cases.append(case("ties n+=200 n-=800", pos, neg, 0.1, 0.05, 0.05, 0.05, rng))

    out = {
        "generator": "scripts/make_conformal_golden.py",
        "versions": {"mapie": md.version("mapie"), "crepes": md.version("crepes"), "numpy": np.__version__},
        "cases": cases,
    }
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
