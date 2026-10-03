import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { exceedanceTest } from '../src/index.ts';

const golden = JSON.parse(readFileSync(new URL('./fixtures/drift-golden.json', import.meta.url), 'utf8')) as {
  versions: Record<string, string>;
  exceedance: Array<{ count: number; n: number; bound: number; p_value: number }>;
};

test(`scipy ${golden.versions.scipy}: exceedance test p-value is the exact binomial tail`, () => {
  for (const c of golden.exceedance) assert.ok(Math.abs(exceedanceTest(c.count, c.n, c.bound).pValue - c.p_value) < 1e-10, `exceedance ${JSON.stringify(c)}`);
});

test('exceedanceTest validates inputs and alerts above the bound', () => {
  assert.throws(() => exceedanceTest(5, 3, 0.1), /0 <= count <= n/);
  assert.throws(() => exceedanceTest(1, 3, 1.5), /bound/);
  const r = exceedanceTest(30, 1000, 0.01);
  assert.equal(r.alert, true);
  assert.ok(r.lower > 0.01);
  assert.equal(exceedanceTest(10, 1000, 0.01).alert, false);
});
