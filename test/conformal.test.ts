import assert from 'node:assert/strict';
import { test } from 'node:test';

import { conformalLowerThreshold, conformalRank, conformalUpperThreshold, minimumSamples, nextUp } from '../src/index.ts';

function rng(seed: number) {
  let s = seed;
  return () => ((s = (s * 1664525 + 1013904223) % 2 ** 32) / 2 ** 32);
}

test('sample sufficiency matches the exact binomial tables', () => {
  assert.deepEqual([0.1, 0.05, 0.02, 0.01].map((a) => minimumSamples(a)), [9, 19, 49, 99]);
  assert.deepEqual([0.1, 0.05, 0.02, 0.01].map((a) => minimumSamples(a, 0.05)), [29, 59, 149, 299]);
  assert.deepEqual([59, 150, 500, 1000].map((n) => conformalRank(n, 0.05, 0.05)), [0, 2, 16, 38]);
  assert.equal(conformalRank(58, 0.05, 0.05), -1);
  assert.equal(conformalRank(19, 0.05), 0, 'expected: ⌊α(n + 1)⌋ - 1');
  assert.equal(conformalRank(18, 0.05), -1);
  assert.throws(() => conformalRank(10, 0), /alpha must be strictly between 0 and 1/);
  assert.throws(() => conformalRank(10, 0.1, 1), /delta must be strictly between 0 and 1/);
  assert.throws(() => conformalRank(1.5, 0.1), /non-negative integer/);
});

test('thresholds are order statistics: from below and from above', () => {
  const pos = [0.9, 0.2, 0.5, 0.7, 0.8, 0.6, 0.95, 0.85, 0.75, 0.65];
  assert.equal(conformalLowerThreshold(pos, 0.5), 0.7, 'k = ⌊0.5 · 11⌋ - 1 = 4 scores may fall below: the 5th smallest');
  assert.equal(conformalLowerThreshold(pos, 0.05), null);
  const neg = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95];
  const t = conformalUpperThreshold(neg, 0.5)!;
  assert.equal(t, nextUp(0.6), 'four may reach it, so it sits just above the 5th largest');
  assert.equal(neg.filter((p) => p >= t).length, 4);
  assert.equal(conformalUpperThreshold(neg, 0.05), null);
});

test('nextUp is the next double up', () => {
  assert.equal(nextUp(0.6) - 0.6, 2 ** -53, 'one unit in the last place, for 0.5 <= v < 1');
  assert.equal(nextUp(1) - 1, Number.EPSILON);
  assert.equal(nextUp(0), Number.MIN_VALUE);
  assert.ok(nextUp(-1) > -1 && nextUp(-1) < -0.9999999999999998);
  assert.equal(nextUp(Infinity), Infinity);
  assert.ok(Number.isNaN(nextUp(NaN)));
});

test('coverage: guarantees fail at most δ of the time (PAC) or hold on average (expected) across resampled calibration sets', () => {
  // Uniform scores: a threshold t leaves exactly t of future scores below it and 1 - t at or above.
  const rand = rng(42);
  const runs = 400, n = 150, alpha = 0.05, delta = 0.05;
  let lowerFails = 0, upperFails = 0, expectedBelow = 0, expectedAbove = 0;
  for (let r = 0; r < runs; r++) {
    const scores = Array.from({ length: n }, rand);
    if (conformalLowerThreshold(scores, alpha, delta)! > alpha) lowerFails++;
    if (1 - conformalUpperThreshold(scores, alpha, delta)! > alpha) upperFails++;
    expectedBelow += conformalLowerThreshold(scores, alpha)!;
    expectedAbove += 1 - conformalUpperThreshold(scores, alpha)!;
  }
  const slack = 3 * Math.sqrt((delta * (1 - delta)) / runs); // simulation error
  assert.ok(lowerFails / runs <= delta + slack, `lower guarantee failed in ${lowerFails}/${runs} runs`);
  assert.ok(upperFails / runs <= delta + slack, `upper guarantee failed in ${upperFails}/${runs} runs`);
  assert.ok(expectedBelow / runs <= alpha + 0.005, `mean share below ${expectedBelow / runs}`);
  assert.ok(expectedAbove / runs <= alpha + 0.005, `mean share above ${expectedAbove / runs}`);
});
