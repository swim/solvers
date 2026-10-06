/**
 * The TypeScript solvers must reproduce scikit-learn (test/fixtures/sklearn-golden.json, from
 * scripts/make_golden.py). Logistic regression is convex, so two correct solvers agree up to
 * solver tolerance; isotonic regression is a deterministic algorithm port, so it must match to
 * float precision.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import {
  decisionFunction,
  effectiveWeights,
  fitIsotonic,
  fitLogistic,
  fitPlatt,
  predictIsotonic,
  sigmoid,
} from '../src/index.ts';

const golden = JSON.parse(readFileSync(new URL('./fixtures/sklearn-golden.json', import.meta.url), 'utf8'));

const COEF_TOL = 1e-6;
const PROBA_TOL = 1e-8;
const EXACT_TOL = 1e-12;

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  assert.equal(a.length, b.length);
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
}

for (const c of golden.logistic) {
  test(`logistic regression matches scikit-learn: ${c.name}`, () => {
    const model = fitLogistic(c.X, c.y, {
      C: c.C,
      classWeight: c.class_weight,
      sampleWeight: c.sample_weight ?? undefined,
    });
    const coefDiff = maxAbsDiff(model.coef, c.coef);
    const interceptDiff = Math.abs(model.intercept - c.intercept);
    const proba = c.X.map((x: number[]) => sigmoid(decisionFunction(model, x)));
    const probaDiff = maxAbsDiff(proba, c.proba);
    assert.ok(coefDiff < COEF_TOL, `coef differs by ${coefDiff}`);
    assert.ok(interceptDiff < COEF_TOL, `intercept differs by ${interceptDiff}`);
    assert.ok(probaDiff < PROBA_TOL, `probabilities differ by ${probaDiff}`);
    assert.ok(model.iterations < 30, `took ${model.iterations} Newton iterations`);
    assert.ok(model.converged, 'Newton-decrement stopping test met');
  });
}

test('Platt scaling (prevalence-weighted) matches scikit-learn', () => {
  const p = golden.platt;
  const platt = fitPlatt(p.logits, p.y, p.sample_weight);
  assert.equal(platt.converged, true, 'fitPlatt reports convergence');
  assert.ok(Math.abs(platt.a - p.a) < COEF_TOL, `a differs: ${platt.a} vs ${p.a}`);
  assert.ok(Math.abs(platt.c - p.c) < COEF_TOL, `c differs: ${platt.c} vs ${p.c}`);
});

for (const key of ['isotonic', 'isotonic_zero_weight']) {
  test(`weighted isotonic regression reproduces scikit-learn thresholds and predictions: ${key}`, () => {
    const iso = golden[key];
    const model = fitIsotonic(iso.x, iso.y, iso.sample_weight, { yMin: 0, yMax: 1 });
    assert.equal(model.x.length, iso.x_thresholds.length, 'number of thresholds');
    assert.ok(maxAbsDiff(model.x, iso.x_thresholds) < EXACT_TOL, 'x thresholds');
    assert.ok(maxAbsDiff(model.y, iso.y_thresholds) < EXACT_TOL, 'y thresholds');
    const predictions = iso.grid.map((v: number) => predictIsotonic(model, v));
    assert.ok(maxAbsDiff(predictions, iso.grid_predictions) < EXACT_TOL, 'grid predictions (incl. clipping)');
  });
}

test('isotonic: a zero-weight sample at a unique x is dropped, as in scikit-learn', () => {
  // scikit-learn 1.9.1: X_thresholds_ [0, 2, 3], y_thresholds_ [0, 0, 1]
  assert.deepEqual(fitIsotonic([0, 1, 2, 3], [0, 1, 0, 1], [1, 0, 1, 1]), { x: [0, 2, 3], y: [0, 0, 1] });
});

test('logistic: converged is false when maxIter stops the fit early', () => {
  const model = fitLogistic([[-3], [-2], [-1], [1], [2], [3]], [0, 0, 0, 1, 1, 1], { maxIter: 1 });
  assert.equal(model.iterations, 1);
  assert.equal(model.converged, false);
});

test('input validation', () => {
  assert.throws(() => fitLogistic([[1], [2]], [0, 2]), /0\/1/);
  assert.throws(() => fitLogistic([[1], [2]], [1, 1], { classWeight: 'balanced' }), /both classes/);
  assert.throws(() => fitLogistic([[1, 2], [3]], [0, 1]), /same length/);
  const X = [[0], [1], [2], [3]];
  const y = [0, 0, 1, 1];
  assert.throws(() => fitLogistic(X, y, { sampleWeight: [1, 1] }), /sampleWeight has length 2, expected 4/);
  assert.throws(() => fitLogistic(X, y, { sampleWeight: [1, -5, 1, 1] }), /non-negative/);
  assert.throws(() => fitLogistic(X, y, { sampleWeight: [1, NaN, 1, 1] }), /finite/);
  assert.throws(() => fitLogistic(X, y, { sampleWeight: [0, 0, 0, 0] }), /all be zero/);
  assert.throws(() => effectiveWeights([0, 2, 1], { classWeight: 'balanced' }), /0\/1/);
  assert.throws(() => decisionFunction({ coef: [1, 2], intercept: 0 }, [1]), /expects 2/);
  assert.throws(() => fitIsotonic([0, 1], [0, 1], [1]), /sampleWeight has length/);
  assert.throws(() => fitIsotonic([0, 1], [0, 1], [1, -1]), /non-negative/);
  assert.throws(() => fitIsotonic([0, 1], [0, 1], [0, 0]), /all be zero/);
  assert.throws(() => fitIsotonic([0, NaN, 2], [0, 1, 0]), /finite/);
});
