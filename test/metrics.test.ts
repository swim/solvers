import assert from 'node:assert/strict';
import { test } from 'node:test';

import { cohenKappa, ece, prevalenceWeights, wilson } from '../src/index.ts';

const close = (a: number, b: number, tol = 1e-12) => assert.ok(Math.abs(a - b) < tol, `${a} vs ${b}`);

test('Wilson interval (146/150 -> 0.933-0.990)', () => {
  const [lo, hi] = wilson(146, 150);
  close(lo, 0.9334, 1e-4);
  close(hi, 0.9896, 1e-4);
  assert.deepEqual(wilson(0, 0), [0, 1]);
});

test('prevalence weights put the target share of weight on positives', () => {
  const y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0];
  const w = prevalenceWeights(y, 0.05);
  const total = w.reduce((s, v) => s + v, 0);
  close(w.filter((_, i) => y[i] === 1).reduce((s, v) => s + v, 0) / total, 0.05);
  assert.throws(() => prevalenceWeights([1, 1], 0.1), /one class/);
  assert.throws(() => prevalenceWeights([0, 1], 1.5), /strictly between/);
  assert.throws(() => prevalenceWeights([0, 1], 0), /strictly between/);
  assert.throws(() => prevalenceWeights([0, 2], 0.1), /0\/1/);
});

test('ECE: perfect calibration is 0; numpy-style bin edges', () => {
  close(ece([0.25, 0.25, 0.25, 0.25], [1, 0, 0, 0], [1, 1, 1, 1]).ece, 0);
  // 0.3 falls in bin 2 (edges[3] = 3*0.1 = 0.30000000000000004 > 0.3), as numpy.digitize does
  assert.equal(ece([0.3], [1], [1]).reliability[0].bin, '0.2-0.3');
  assert.equal(ece([1.0], [1], [1]).reliability[0].bin, '0.9-1.0');
  close(ece([0.9, 0.9], [0, 0], [1, 1]).ece, 0.9);
  // left-closed (np.digitize): exactly-on-edge values go to the upper bin
  assert.equal(ece([0.5], [1], [1]).reliability[0].bin, '0.5-0.6');
  assert.equal(ece([0], [0], [1]).reliability[0].bin, '0.0-0.1');
  // labels carry enough decimals for the bin count
  assert.equal(ece([0.16], [1], [1], 20).reliability[0].bin, '0.15-0.20');
  assert.equal(ece([0.5], [1], [1], 4).reliability[0].bin, '0.50-0.75');
  assert.throws(() => ece([0.5], [1], [0]), /total weight/);
  assert.throws(() => ece([0.5, 0.6], [1], [1, 1]), /same length/);
  assert.throws(() => ece([0.5], [1], [1], 0), /positive integer/);
});

test("Cohen's kappa (textbook: observed 0.7, expected 0.5 -> 0.4)", () => {
  const a = [...Array(25).fill('y'), ...Array(25).fill('n')];
  const b = [...Array(20).fill('y'), ...Array(5).fill('n'), ...Array(10).fill('y'), ...Array(15).fill('n')];
  close(cohenKappa(a, b), 0.4);
  close(cohenKappa(['a', 'b'], ['a', 'b']), 1);
  assert.equal(cohenKappa(['a', 'a'], ['a', 'a']), 1); // scikit-learn returns NaN here
  assert.throws(() => cohenKappa(['a', 'b', 'a'], ['a', 'b']), /same length/);
  assert.throws(() => cohenKappa([], []), /non-empty/);
});

test('binomial CDF and exact Clopper-Pearson upper bound', async () => {
  const { binomialCdf, clopperPearsonUpper } = await import('../src/index.ts');
  close(binomialCdf(0, 10, 0.1), 0.9 ** 10);
  close(binomialCdf(2, 5, 0.5), 0.5, 1e-12); // (1 + 5 + 10) / 32
  close(binomialCdf(5, 5, 0.3), 1);
  close(clopperPearsonUpper(0, 1000), 1 - 0.05 ** (1 / 1000), 1e-12);
  // one-sided 95%: scipy.stats.beta.ppf(0.95, 4, 97) = 0.0757 for k=3, n=100
  close(clopperPearsonUpper(3, 100), 0.07571, 1e-4);
  assert.equal(clopperPearsonUpper(0, 0), 1);
});
