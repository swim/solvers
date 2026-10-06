# API

Every export, what it does, and what it is checked against.

| Export | What | Matches scikit-learn |
|---|---|---|
| `fitLogistic`, `decisionFunction`, `sigmoid`, `effectiveWeights` | Binary logistic regression, L2, per-sample weights, `classWeight: 'balanced'`. Damped Newton (IRLS) with a Cholesky solve | `LogisticRegression(penalty="l2")`: same objective, same optimum |
| `fitPlatt`, `predictPlatt` | Platt scaling: a weighted logistic regression on the classifier's logit | `LogisticRegression(C=1e6)` on the logit (not `CalibratedClassifierCV`; see below) |
| `fitIsotonic`, `predictIsotonic` | Weighted, non-decreasing isotonic regression, ported step for step | `IsotonicRegression(out_of_bounds="clip", y_min, y_max)` |
| `wilson` | Wilson score interval for a proportion | |
| `prevalenceWeights` | Reweights an enriched sample so positives carry a target prevalence | |
| `ece` | Weighted expected calibration error plus a reliability table | numpy `linspace` / `digitize` binning (not `calibration_curve`; see below) |
| `cohenKappa` | Inter-annotator agreement | `cohen_kappa_score`, except the degenerate case below |
| `binomialCdf`, `clopperPearsonUpper` | Exact binomial CDF and one-sided Clopper-Pearson upper bound | |
| `conformalLowerThreshold`, `conformalUpperThreshold`, `conformalRank`, `minimumSamples` | Distribution-free thresholds from order statistics: at most a share α of future scores below (or at and above) the threshold, with probability 1 − δ (PAC) or on average (no δ). E.g. a recall threshold from positives' scores, a false-alarm threshold from negatives'. `minimumSamples` is the fewest scores for any guarantee | MAPIE `BinaryClassificationController` (Learn Then Test, fixed sequence) and crepes class-conditional p-values |
| `nextUp` | The next double above a value, so `score >= nextUp(v)` excludes `v` | |
| `htTotal`, `stratifiedRatio`, `stratifiedBootstrap`, `kishEffectiveN`, `weightedQuantile` | Design-based estimation for stratified simple random samples: Horvitz–Thompson totals, ratios with linearised variance and finite-population correction (a stratum left with one sampled unit pools its variance with the partner stratum of closest sampling weight - collapsed strata - and is listed in `collapsed`), Rao–Wu bootstrap replicate weights, Kish effective size, weighted quantiles | samplics `TaylorEstimator` (totals, ratios, standard errors) |
| `designRiskThreshold` | A recall threshold from a stratified sample whose miss rate is ≤ α with probability 1 − δ. `exact` (default): per-stratum Clopper–Pearson bounds, valid but conservative with many strata. `linearised` / `bootstrap`: approximate. They under-cover when a heavily weighted stratum yields few sampled positives, typically because high-score strata were over-sampled to find positives. Such "thin" strata are reported in `warnings`; `failOnThinStrata` turns the warning into infeasibility. A stratum with one sampled unit (e.g. after reviewer skips) no longer throws: `exact` needs no variance, `linearised` pools it with a partner stratum and warns; only `bootstrap` still needs two | |
| `designPrecisionThreshold` | A precision threshold from a stratified sample: the loosest of a fixed, strictest-first candidate list whose precision lower bound reaches the target. `linearised`, or `exact` (rarely feasible: it must allow for unseen false positives in every stratum) | |
| `coxTest` | Cox's recalibration test: y ~ a + b·logit p, likelihood-ratio test of a = 0, b = 1 (weights rescaled to their Kish size). `converged` is false under separation, when the p-value means nothing | scikit-learn's unpenalised `LogisticRegression` |
| `exceedanceTest` | Exact binomial test that a live share of scores at or above a threshold exceeds a bound | scipy |
| `clopperPearsonLower`, `normalQuantile`, `seededRandom` | Exact binomial lower bound; standard normal quantile; a seeded PRNG | |

## Why exact solvers

The L2 logistic objective is strictly convex, so it has exactly one optimum. Any correct solver
reaches the same coefficients, which makes the results checkable against a reference
implementation. Newton's method converges to that optimum in a handful of iterations; first-order
solvers often stop short.

## Limits

- Binary only (use one-vs-rest for multi-label), L2 only, dense features.
- Each Newton step costs O(n·d²) + O(d³). That is fine up to a few thousand features.
- No Node built-ins and no runtime dependencies: runs on serverless, edge and browser runtimes (checked by a test that walks the import graph).
