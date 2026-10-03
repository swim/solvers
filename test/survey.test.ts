import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import {
  designRiskThreshold,
  htTotal,
  kishEffectiveN,
  normalQuantile,
  seededRandom,
  stratifiedBootstrap,
  stratifiedRatio,
  weightedQuantile,
} from '../src/index.ts';

interface GoldenCase {
  name: string; strata: string[]; inclusion_probs: number[]; stratum_sizes: Record<string, number>;
  y: number[]; scores: number[]; t: number;
  total: number; total_se: number; miss_rate: number; miss_rate_se: number; precision: number; precision_se: number;
}
const golden = JSON.parse(readFileSync(new URL('./fixtures/survey-golden.json', import.meta.url), 'utf8')) as { versions: Record<string, string>; cases: GoldenCase[] };
const close = (a: number, b: number, tol: number, what: string) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${what}: ${a} vs ${b}`);

for (const c of golden.cases) {
  test(`samplics ${golden.versions.samplics}: ${c.name}`, () => {
    const design = { inclusionProbs: c.inclusion_probs, strata: c.strata, stratumSizes: c.stratum_sizes };
    close(htTotal(c.y, c.inclusion_probs), c.total, 1e-8, 'total');
    // A total's linearised SE is the ratio's with denominator total 1: pass den = π·ΣN so B̂ = 1.
    const N = Object.values(c.stratum_sizes).reduce((a, b) => a + b, 0);
    const asTotal = stratifiedRatio({ ...design, num: c.y, den: c.y.map(() => 1 / N) });
    close(asTotal.se, c.total_se, 1e-8, 'total SE');
    const missed = c.y.map((v, i) => (v === 1 && c.scores[i] < c.t ? 1 : 0));
    const miss = stratifiedRatio({ ...design, num: missed, den: c.y });
    close(miss.estimate, c.miss_rate, 1e-8, 'miss rate');
    close(miss.se, c.miss_rate_se, 1e-8, 'miss rate SE');
    const fired = c.scores.map((s) => (s >= c.t ? 1 : 0));
    const precision = stratifiedRatio({ ...design, num: c.y.map((v, i) => v * fired[i]), den: fired });
    close(precision.estimate, c.precision, 1e-8, 'precision');
    close(precision.se, c.precision_se, 1e-8, 'precision SE');
  });
}

test('estimators validate their inputs', () => {
  assert.throws(() => htTotal([1, 2], [0.5]), /inclusionProbs has 1 entries, expected 2/);
  assert.throws(() => htTotal([1], [0]), /must be in \(0, 1\]/);
  assert.throws(() => htTotal([NaN], [0.5]), /not finite/);
  const design = { inclusionProbs: [0.1, 0.1, 0.2], strata: ['a', 'a', 'b'], stratumSizes: { a: 20, b: 5 } };
  assert.throws(() => stratifiedRatio({ ...design, num: [1, 0, 1], den: [1, 1, 1] }), /stratum b has 1 sampled unit/);
  assert.throws(() => stratifiedRatio({ ...design, inclusionProbs: [0.1, 0.2, 0.2], num: [1, 0, 1], den: [1, 1, 1] }), /stratified simple random sampling only/);
  assert.throws(() => stratifiedRatio({ inclusionProbs: [0.5, 0.5], strata: ['a', 'a'], stratumSizes: { a: 4 }, num: [1, 1], den: [0, 0] }), /denominator total is 0/);
});

test('Kish effective n, weighted quantiles and the normal quantile', () => {
  assert.equal(kishEffectiveN([1, 1, 1, 1]), 4);
  assert.equal(kishEffectiveN([3, 1]), 16 / 10);
  assert.equal(weightedQuantile([3, 1, 2], [1, 1, 1], 0.5), 2);
  assert.equal(weightedQuantile([1, 2, 3], [1, 1, 8], 0.2), 2);
  assert.equal(weightedQuantile([1, 2, 2, 3], [1, 1, 1, 1], 0.3), 2, 'ties share the cumulative weight of the tied group');
  assert.equal(weightedQuantile([1, 2, 3], [1, 1, 1], 1), 3);
  assert.ok(Math.abs(normalQuantile(0.95) - 1.6448536269514722) < 1e-8);
  assert.ok(Math.abs(normalQuantile(0.001) + 3.090232306167813) < 1e-8);
  assert.ok(Math.abs(normalQuantile(0.5)) < 1e-12);
});

/** A population with known scores and labels, stratified by score band. */
function population(N: number, seed: number) {
  const rand = seededRandom(seed);
  const beta = (a: number) => { // mean-a skewed score in (0, 1), cheap and deterministic
    const u = rand();
    return a > 0.5 ? 1 - (1 - u) ** 2.5 : u ** 2.5;
  };
  const y: number[] = [], s: number[] = [];
  for (let i = 0; i < N; i++) {
    const pos = rand() < 0.05 ? 1 : 0;
    y.push(pos);
    s.push(beta(pos ? 0.8 : 0.2));
  }
  const band = s.map((v) => (v >= 0.6 ? 'hi' : v >= 0.3 ? 'mid' : 'lo'));
  return { y, s, band };
}

function sample(pop: ReturnType<typeof population>, n: Record<string, number>, rand: () => number) {
  const by: Record<string, number[]> = {};
  pop.band.forEach((b, i) => (by[b] ??= []).push(i));
  const idx: number[] = [], strata: string[] = [], pis: number[] = [];
  const sizes: Record<string, number> = {};
  for (const [b, members] of Object.entries(by)) {
    sizes[b] = members.length;
    const take = Math.min(n[b], members.length);
    const pool = [...members];
    for (let k = 0; k < take; k++) {
      const j = k + Math.floor(rand() * (pool.length - k));
      [pool[k], pool[j]] = [pool[j], pool[k]];
      idx.push(pool[k]); strata.push(b); pis.push(take / members.length);
    }
  }
  return { idx, strata, pis, sizes };
}

test('bootstrap: deterministic for a seed, and close to the linearised SE on a large design', () => {
  const pop = population(40000, 3);
  const d = sample(pop, { hi: 600, mid: 500, lo: 400 }, seededRandom(4));
  const a = stratifiedBootstrap({ inclusionProbs: d.pis, strata: d.strata, replicates: 50, seed: 9 });
  assert.deepEqual(a, stratifiedBootstrap({ inclusionProbs: d.pis, strata: d.strata, replicates: 50, seed: 9 }));
  assert.notDeepEqual(a, stratifiedBootstrap({ inclusionProbs: d.pis, strata: d.strata, replicates: 50, seed: 10 }));

  const y = d.idx.map((i) => pop.y[i]);
  const num = d.idx.map((i) => (pop.y[i] === 1 && pop.s[i] < 0.5 ? 1 : 0));
  const lin = stratifiedRatio({ inclusionProbs: d.pis, strata: d.strata, stratumSizes: d.sizes, num, den: y });
  const reps = stratifiedBootstrap({ inclusionProbs: d.pis, strata: d.strata, replicates: 2000, seed: 1 });
  const rs = reps.map((w) => w.reduce((s, wi, i) => s + wi * num[i], 0) / w.reduce((s, wi, i) => s + wi * y[i], 0));
  const mean = rs.reduce((x, v) => x + v, 0) / rs.length;
  const se = Math.sqrt(rs.reduce((x, v) => x + (v - mean) ** 2, 0) / (rs.length - 1));
  assert.ok(Math.abs(se / lin.se - 1) < 0.05, `bootstrap SE ${se} vs linearised ${lin.se}`);
});

function coverage(n: Record<string, number>, alpha: number, method: 'exact' | 'linearised', runs = 500) {
  const pop = population(60000, 11);
  const posScores = pop.s.filter((_, i) => pop.y[i] === 1).sort((a, b) => a - b);
  const trueMiss = (t: number) => posScores.filter((v) => v < t).length / posScores.length;
  const rand = seededRandom(12);
  let fails = 0, infeasible = 0;
  for (let r = 0; r < runs; r++) {
    const d = sample(pop, n, rand);
    const res = designRiskThreshold({ inclusionProbs: d.pis, strata: d.strata, stratumSizes: d.sizes, y: d.idx.map((i) => pop.y[i]), scores: d.idx.map((i) => pop.s[i]), alpha, delta: 0.05, method });
    if (!res.feasible) infeasible++;
    else if (trueMiss(res.threshold) > alpha) fails++;
  }
  return { fails, infeasible, runs };
}
// In this population 13% of positives score in the low band, where they are 1% of traffic.
const specDesign = { hi: 500, mid: 300, lo: 200 };
const loOversampled = { hi: 400, mid: 600, lo: 4000 };
const slack = 3 * Math.sqrt((0.05 * 0.95) / 500);

test('designRiskThreshold (exact): the true miss rate exceeds α in at most δ of 500 sampled designs', () => {
  const r = coverage(loOversampled, 0.2, 'exact');
  assert.equal(r.infeasible, 0);
  assert.ok(r.fails / r.runs <= 0.05 + slack, `true miss rate exceeded α in ${r.fails}/${r.runs} runs`);
});

test('designRiskThreshold (exact): infeasible rather than overconfident when a heavily weighted stratum is barely sampled', () => {
  const r = coverage(specDesign, 0.1, 'exact', 100);
  assert.equal(r.infeasible, r.runs, 'about 2 sampled positives stand for 13% of all positives: no 90% guarantee is honest');
});

test('designRiskThreshold (linearised): under-covers in that same design - why it is not the default', () => {
  const r = coverage(specDesign, 0.1, 'linearised');
  assert.ok(r.fails / r.runs > 2 * 0.05, `expected under-coverage, got ${r.fails}/${r.runs}`);
  const ok = coverage(loOversampled, 0.2, 'linearised');
  assert.ok(ok.fails / ok.runs <= 0.05 + slack, 'adequate when every stratum yields enough positives');
});

test('designRiskThreshold (linearised): thin-strata warnings flag the under-allocated design; failOnThinStrata turns them into infeasibility', () => {
  const pop = population(60000, 11);
  const posScores = pop.s.filter((_, i) => pop.y[i] === 1).sort((a, b) => a - b);
  const trueMiss = (t: number) => posScores.filter((v) => v < t).length / posScores.length;
  const rand = seededRandom(12);
  let warned = 0, failsWarned = 0, fails = 0, guardedFails = 0, guardedFeasible = 0;
  const runs = 300;
  for (let r = 0; r < runs; r++) {
    const d = sample(pop, specDesign, rand);
    const args = { inclusionProbs: d.pis, strata: d.strata, stratumSizes: d.sizes, y: d.idx.map((i) => pop.y[i]), scores: d.idx.map((i) => pop.s[i]), alpha: 0.1, delta: 0.05, method: 'linearised' as const };
    const res = designRiskThreshold(args);
    const failed = res.feasible && trueMiss(res.threshold) > 0.1;
    if (res.warnings?.length) warned++;
    if (failed) { fails++; if (res.warnings?.length) failsWarned++; }
    const guarded = designRiskThreshold({ ...args, failOnThinStrata: true });
    if (guarded.feasible) { guardedFeasible++; if (trueMiss(guarded.threshold) > 0.1) guardedFails++; }
  }
  assert.ok(fails > 0.1 * runs, `${fails} failures`);
  assert.ok(failsWarned >= 0.9 * fails, `the warning fires on ${failsWarned} of ${fails} failing runs`);
  assert.ok(warned >= 0.9 * runs, 'this design is flagged almost every time');
  assert.ok(guardedFails / runs <= 0.05 + slack, `failOnThinStrata: ${guardedFails} failures in ${guardedFeasible} feasible runs`);
  const exact = designRiskThreshold({ inclusionProbs: [0.5, 0.5, 0.5, 0.5], strata: ['a', 'a', 'a', 'a'], stratumSizes: { a: 8 }, y: [1, 1, 0, 0], scores: [0.9, 0.8, 0.1, 0.2], alpha: 0.5, delta: 0.05 });
  assert.deepEqual(exact.feasible ? exact.warnings : exact.warnings ?? [], [], 'exact needs no warnings');
});

test('designRiskThreshold: infeasible with too few positives, refuses positives-only input, bootstrap agrees', () => {
  const few = { inclusionProbs: [0.5, 0.5, 0.5, 0.5], strata: ['a', 'a', 'a', 'a'], stratumSizes: { a: 8 }, y: [1, 1, 0, 0], scores: [0.9, 0.8, 0.1, 0.2], alpha: 0.05, delta: 0.05 };
  const res = designRiskThreshold(few);
  assert.equal(res.feasible, false);
  assert.match((res as { reason: string }).reason, /even at the lowest positive score/);
  assert.throws(() => designRiskThreshold({ ...few, y: [1, 1, 1, 1] }), /not only positives/);

  const pop = population(40000, 21);
  const d = sample(pop, loOversampled, seededRandom(22));
  const base = { inclusionProbs: d.pis, strata: d.strata, stratumSizes: d.sizes, y: d.idx.map((i) => pop.y[i]), scores: d.idx.map((i) => pop.s[i]), alpha: 0.2, delta: 0.05 };
  const lin = designRiskThreshold({ ...base, method: 'linearised' }), boot = designRiskThreshold({ ...base, method: 'bootstrap', replicates: 500, seed: 3 });
  assert.ok(lin.feasible && boot.feasible);
  if (lin.feasible && boot.feasible) {
    assert.equal(boot.method, 'bootstrap');
    assert.ok(Math.abs(lin.missRateUpper - boot.missRateUpper) < 0.03, `${lin.missRateUpper} vs ${boot.missRateUpper}`);
    assert.ok(lin.missRateUpper <= 0.2 && lin.missRateEstimate <= lin.missRateUpper);
    // The returned threshold matches a direct linearised computation at it.
    const num = base.y.map((v, i) => (v === 1 && base.scores[i] < lin.threshold ? 1 : 0));
    assert.ok(Math.abs(stratifiedRatio({ ...base, num, den: base.y }).estimate - lin.missRateEstimate) < 1e-12);
  }
  const exact = designRiskThreshold(base);
  assert.equal(exact.feasible ? exact.guarantee : null, 'design-exact');
  if (exact.feasible && lin.feasible) assert.ok(exact.threshold <= lin.threshold, 'exact is the more conservative');
});
