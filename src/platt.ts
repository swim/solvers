/**
 * Platt scaling: a (nearly unregularised) weighted logistic regression on a classifier's logit,
 * giving p = σ(a·logit + c). Matches scikit-learn's LogisticRegression(C=1e6) fitted on the logit.
 */
import { fitLogistic, sigmoid } from './logistic.ts';

export interface PlattModel {
  a: number;
  c: number;
}

export function fitPlatt(logits: ArrayLike<number>, y: ArrayLike<number>, sampleWeight?: ArrayLike<number>): PlattModel {
  const model = fitLogistic(Array.from(logits, (z) => [z]), y, { C: 1e6, sampleWeight });
  return { a: model.coef[0], c: model.intercept };
}

export function predictPlatt(platt: PlattModel, logit: number): number {
  return sigmoid(platt.a * logit + platt.c);
}
