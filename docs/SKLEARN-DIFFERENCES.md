# Differences from scikit-learn


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
