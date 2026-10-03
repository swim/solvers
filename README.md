# @liquidau/solvers

Exact solvers for small, dense binary classification problems (such as heads on top of text
embeddings), in TypeScript and **verified against scikit-learn**, plus the metrics you need to
calibrate and evaluate them. One runtime dependency: [`ml-matrix`](https://github.com/mljs/matrix).

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

```ts
import { fitLogistic, decisionFunction, fitPlatt, predictPlatt, prevalenceWeights } from '@liquidau/solvers';

const model = fitLogistic(Xtrain, ytrain, { C: 1, classWeight: 'balanced' });
const logits = Xcal.map((x) => decisionFunction(model, x));
// Calibrate to the class balance you expect in production, not the enriched training mix.
const platt = fitPlatt(logits, ycal, prevalenceWeights(ycal, 0.01));
const p = predictPlatt(platt, decisionFunction(model, x));
```

## Why exact solvers

The L2 logistic objective is strictly convex, so it has exactly one optimum. Any correct solver
reaches the same coefficients, which makes the results checkable against a reference
implementation. Newton's method converges to that optimum in a handful of iterations; first-order
solvers often stop short.

## Verification

`npm test` checks every solver against `test/fixtures/sklearn-golden.json`, which scikit-learn's
exact `newton-cholesky` solver generates (`scripts/make_golden.py`). The fixture covers
class-balanced, sample-weighted, strongly regularised and balanced-plus-weighted cases, plus an
isotonic fit with zero-weight samples. The
tolerances are:

- coefficients: below 1e-6
- probabilities: below 1e-8
- isotonic thresholds and predictions: below 1e-12

`test/fixtures/conformal-golden.json` (`scripts/make_conformal_golden.py`) checks the conformal
thresholds against MAPIE and crepes. They match exactly, and every MAPIE p-value is reproduced to
1e-8. One known MAPIE difference: its built-in `recall` risk computes 1 − recall in floating point and
rounds the miss count up, which can count one miss too many (150 × (1 − 149/150) =
1.0000000000000064). In 3 of the 8 cases it lands one rank stricter than the exact test. The
fixture therefore states recall as a miss rate, and records the built-in result for comparison.

At 3,000 examples × 512 features, a fit takes about 10 s over 7 or 8 Newton iterations
(Apple Silicon, Node 20). Almost all of that is forming the Hessian, about 1 s per iteration.

`classWeight: 'balanced'` uses sample-weighted class totals, as current scikit-learn does.

`fitLogistic` returns `converged: true` only when the Newton-decrement stopping test passed. It is
`false` if the fit hit `maxIter` or the line search stalled first.

## Differences from scikit-learn

- **Sample weights.** Negative or non-finite weights throw, as do weights that are all zero.
  scikit-learn's isotonic regression silently drops negative weights, and its logistic regression
  accepts them. Zero weights are dropped from isotonic fits, as in scikit-learn.
- **Platt scaling.** `fitPlatt` fits hard 0/1 targets with a tiny L2 penalty (`C = 1e6`).
  `CalibratedClassifierCV(method="sigmoid")` uses Platt's smoothed targets and no penalty, so its
  coefficients differ slightly.
- **ECE bins.** Bins are closed on the left, as `np.digitize` does, so 0.5 falls in `0.5-0.6`.
  `calibration_curve` closes them on the right and puts 0.5 in `0.4-0.5`. Edges follow `linspace`
  arithmetic, so 0.3 falls in `0.2-0.3` because the computed edge is 0.30000000000000004.
- **Cohen's kappa.** When both raters use one identical label throughout, `cohenKappa` returns 1.
  `cohen_kappa_score` returns NaN.

## Limits

- Binary only (use one-vs-rest for multi-label), L2 only, dense features.
- Each Newton step costs O(n·d²) + O(d³). That is fine up to a few thousand features.

## Develop

```bash
npm install          # TypeScript, tsx and @types/node are devDependencies
npm test             # runs the .ts tests through tsx, so Node 20 works
npm run typecheck
npm run build        # dist/ (ESM + .d.ts)

# Regenerating the golden fixtures needs Python (development only):
python3 -m venv .venv && .venv/bin/pip install numpy scikit-learn mapie crepes
.venv/bin/python scripts/make_golden.py
.venv/bin/python scripts/make_conformal_golden.py
```

## License

MIT. See [LICENSE](LICENSE).

The isotonic regression in `src/isotonic.ts` is ported from scikit-learn and keeps its
BSD-3-Clause notice in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
