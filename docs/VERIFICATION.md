# Verification


`npm test` checks every solver against `test/fixtures/sklearn-golden.json`, which scikit-learn's
exact `newton-cholesky` solver generates (`scripts/make_golden.py`). The fixture covers
class-balanced, sample-weighted, strongly regularised and balanced-plus-weighted cases, plus an
isotonic fit with zero-weight samples. The
tolerances are:

- coefficients: below 1e-6
- probabilities: below 1e-8
- isotonic thresholds and predictions: below 1e-12

`test/fixtures/survey-golden.json` (`scripts/make_survey_golden.py`) checks the design-based
estimators against samplics' `TaylorEstimator`: totals, ratios and their linearised standard errors
match to 1e-8. Note that samplics' `fpc` argument is the multiplier 1 − n/N, not the sampling
fraction. `designRiskThreshold`'s coverage is checked by simulation over 500 designs. In a population
where 13% of positives sit in a low-score stratum that holds 1% of traffic, a 200-item sample of that
stratum makes `exact` infeasible, which is the honest answer. `linearised` exceeds α in 20% of runs at
δ = 5%. When that stratum is oversampled, both hold.

`test/fixtures/drift-golden.json` and `test/fixtures/calibration-golden.json` check `exceedanceTest`
against scipy and Cox's test against scikit-learn (coefficients to 1e-6, p-values to 1e-8).

`test/fixtures/conformal-golden.json` (`scripts/make_conformal_golden.py`) checks the conformal
thresholds against MAPIE and crepes. They match exactly, and every MAPIE p-value is reproduced to
1e-8. One known MAPIE difference: its built-in `recall` risk computes 1 − recall in floating point and
rounds the miss count up, which can count one miss too many (150 × (1 − 149/150) =
1.0000000000000064). In 3 of the 8 cases it lands one rank stricter than the exact test. The
fixture therefore states recall as a miss rate, and records the built-in result for comparison.

At 3,000 examples × 768 features, a fit takes about 6 s over 7 Newton iterations (Apple Silicon,
Node 20); at 616 × 768, about 2 s. X is copied once into a contiguous typed array, the Hessian is
accumulated four samples per pass, and the Cholesky factorisation runs in place.

`classWeight: 'balanced'` uses sample-weighted class totals, as current scikit-learn does.

`fitLogistic` returns `converged: true` only when the Newton-decrement stopping test passed. It is
`false` if the fit hit `maxIter` or the line search stalled first.

## Regenerating the fixtures

The fixtures are checked in, so tests need no Python. To regenerate them:

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scikit-learn mapie crepes samplics
.venv/bin/python scripts/make_golden.py
.venv/bin/python scripts/make_conformal_golden.py
.venv/bin/python scripts/make_survey_golden.py
.venv/bin/python scripts/make_drift_golden.py
.venv/bin/python scripts/make_calibration_golden.py   # needs scikit-learn
```
