/**
 * Platt scaling: a (nearly unregularised) weighted logistic regression on a classifier's logit,
 * giving p = σ(a·logit + c). Matches scikit-learn's LogisticRegression(C=1e6) fitted on the logit.
 */
import { fitLogistic, sigmoid } from './logistic.ts';

export interface PlattModel {
  a: number;
  c: number;
}

/**
 * `converged` is false when Newton's method stopped without meeting its tolerance - typically
 * (quasi-)separation, where the logits split the classes and the nearly unregularised slope grows
 * without bound. Store only `a` and `c`; check `converged` before trusting the fit.
 */
export function fitPlatt(logits: ArrayLike<number>, y: ArrayLike<number>, sampleWeight?: ArrayLike<number>): PlattModel & { converged: boolean } {
  const model = fitLogistic(Array.from(logits, (z) => [z]), y, { C: 1e6, sampleWeight });
  return { a: model.coef[0], c: model.intercept, converged: model.converged };
}

export function predictPlatt(platt: PlattModel, logit: number): number {
  return sigmoid(platt.a * logit + platt.c);
}
