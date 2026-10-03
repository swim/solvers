import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { coxTest } from '../src/index.ts';

const golden = JSON.parse(readFileSync(new URL('./fixtures/calibration-golden.json', import.meta.url), 'utf8')) as {
  versions: Record<string, string>;
  cases: Array<{ name: string; p: number[]; y: number[]; w: number[]; intercept: number; slope: number; lr: number; p_value: number }>;
};

for (const c of golden.cases) {
  test(`scikit-learn ${golden.versions['scikit-learn']}: Cox recalibration test, ${c.name}`, () => {
    const r = coxTest(c.p, c.y, c.w);
    assert.ok(Math.abs(r.intercept - c.intercept) < 1e-6, `intercept ${r.intercept} vs ${c.intercept}`);
    assert.ok(Math.abs(r.slope - c.slope) < 1e-6, `slope ${r.slope} vs ${c.slope}`);
    assert.ok(Math.abs(r.lr - c.lr) < 1e-6 * Math.max(1, c.lr), `LR ${r.lr} vs ${c.lr}`);
    assert.ok(Math.abs(r.pValue - c.p_value) < 1e-8, `p ${r.pValue} vs ${c.p_value}`);
  });
}

test('coxTest validates inputs', () => {
  assert.throws(() => coxTest([0.5, 1], [1, 0]), /strictly between 0 and 1/);
  assert.throws(() => coxTest([0.5], [2]), /0 or 1/);
  assert.throws(() => coxTest([0.5, 0.4], [1]), /same length/);
});

test('coxTest reports non-convergence under separation', () => {
  const p = [0.01, 0.02, 0.03, 0.97, 0.98, 0.99];
  const r = coxTest(p, [0, 0, 0, 1, 1, 1]);
  assert.equal(r.converged, false, `slope ${r.slope}`);
  assert.equal(coxTest([0.2, 0.3, 0.6, 0.7, 0.4, 0.5], [0, 1, 0, 1, 1, 0]).converged, true);
});
