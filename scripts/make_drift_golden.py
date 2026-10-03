"""
Regenerates test/fixtures/drift-golden.json - reference results for src/drift.ts from scipy: the
binomial upper tail for exceedanceTest. Development only; the fixture is checked in.

  python3 -m venv .venv && .venv/bin/pip install numpy scipy
  .venv/bin/python scripts/make_drift_golden.py
"""

import json
import os

import numpy as np
import scipy
from scipy import stats

OUT = os.path.join(os.path.dirname(__file__), "..", "test", "fixtures", "drift-golden.json")


def main():
    exceed = [{"count": k, "n": n, "bound": b, "p_value": float(stats.binom.sf(k - 1, n, b))} for k, n, b in [(0, 100, 0.01), (3, 100, 0.01), (12, 1000, 0.005), (40, 2000, 0.01), (5, 5000, 0.002)]]
    with open(OUT, "w") as f:
        json.dump({"generator": "scripts/make_drift_golden.py", "versions": {"scipy": scipy.__version__, "numpy": np.__version__}, "exceedance": exceed}, f)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
