/**
 * Cox's recalibration test, for a release gate (ECE stays a reported metric: it has no stated error
 * rate and, for rare classes, is dominated by easy negatives): fit y ~ a + b·logit(p) by weighted
 * logistic regression and test a = 0, b = 1 jointly with a likelihood ratio,
 * 2(ℓ(â, b̂) - ℓ(0, 1)) ~ χ²₂, so p = exp(-LR / 2). Also reports â (calibration-in-the-large) and b̂
 * (calibration slope). Spiegelhalter's Z was evaluated and removed: in simulation it had no power
 * against inflated top probabilities, which this test caught.
 *
 * Weights (e.g. design weights 1/π) are rescaled to sum to their Kish effective size, so the
 * likelihood carries the sample's real information - an approximation for unequal weights.
 */
import { fitLogistic } from './logistic.ts';
import { kishEffectiveN } from './survey.ts';

function prep(p: ArrayLike<number>, y: ArrayLike<number>, w?: ArrayLike<number>) {
  const n = p.length;
  if (y.length !== n || (w && w.length !== n)) throw new Error('p, y and w must be the same length');
  if (!n) throw new Error('need at least one example');
  for (let i = 0; i < n; i++) {
    if (!(p[i] > 0 && p[i] < 1)) throw new Error(`p[${i}] must be strictly between 0 and 1, got ${p[i]}`);
    if (y[i] !== 0 && y[i] !== 1) throw new Error(`y[${i}] must be 0 or 1`);
  }
  const raw = Array.from({ length: n }, (_, i) => (w ? w[i] : 1));
  const total = raw.reduce((a, b) => a + b, 0);
  const scale = kishEffectiveN(raw) / total;
  return raw.map((v) => v * scale);
}

/**
 * `converged` is false when the fit didn't settle - typically (quasi-)separation, where the scores
 * split the classes almost perfectly and the maximum-likelihood slope runs off to infinity; the
 * p-value then means nothing.
 */
export function coxTest(p: ArrayLike<number>, y: ArrayLike<number>, w?: ArrayLike<number>): { intercept: number; slope: number; lr: number; pValue: number; converged: boolean } {
  const wt = prep(p, y, w);
  const z = Array.from(p, (v) => Math.log(v / (1 - v)));
  const model = fitLogistic(z.map((v) => [v]), Array.from(y), { C: 1e10, sampleWeight: wt, tol: 1e-14, maxIter: 200 });
  const ll = (a: number, b: number) => {
    let s = 0;
    for (let i = 0; i < z.length; i++) {
      const eta = a + b * z[i];
      // log σ(eta) for y = 1, log(1 - σ(eta)) for y = 0, without overflow.
      const lp = -(eta > 0 ? Math.log1p(Math.exp(-eta)) : -eta + Math.log1p(Math.exp(eta)));
      s += wt[i] * (y[i] === 1 ? lp : lp - eta);
    }
    return s;
  };
  const lr = Math.max(0, 2 * (ll(model.intercept, model.coef[0]) - ll(0, 1)));
  return { intercept: model.intercept, slope: model.coef[0], lr, pValue: Math.exp(-lr / 2), converged: model.converged };
}

