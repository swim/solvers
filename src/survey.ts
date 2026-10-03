/**
 * Design-based estimators for stratified simple random samples without replacement: what a
 * probability sample of real traffic says about the whole of it.
 *
 *   htTotal              Horvitz-Thompson total Σ y_i / π_i
 *   stratifiedRatio      R̂ = Σ (a_i/π_i) / Σ (b_i/π_i) with its linearised variance
 *                        V̂(R̂) = Σ_h N_h² (1 - f_h) s²_{z,h} / n_h,  z_i = (a_i - R̂ b_i) / D̂
 *   stratifiedBootstrap  Rao-Wu rescaling bootstrap replicate weights (no finite-population correction)
 *   kishEffectiveN       (Σ w)² / Σ w²
 *   weightedQuantile     the smallest value whose cumulative normalised weight reaches q
 *   designRiskThreshold  a recall threshold whose miss rate is <= α with probability 1 - δ: exact
 *                        (conservative) per-stratum bounds by default, or the asymptotic
 *                        linearised / bootstrap bounds, which under-cover when a heavily weighted
 *                        stratum holds few sampled positives
 *
 * Verified against samplics' TaylorEstimator (test/fixtures/survey-golden.json). Within a stratum
 * every unit must have the same inclusion probability, n_h / N_h: these are estimators for
 * stratified simple random sampling, and a mismatch throws rather than estimate something else.
 */
import { clopperPearsonLower, clopperPearsonUpper } from './metrics.ts';

function checkFinite(name: string, values: ArrayLike<number>): void {
  for (let i = 0; i < values.length; i++) if (!Number.isFinite(values[i])) throw new Error(`${name}[${i}] is not finite (${values[i]})`);
}

function checkProbs(inclusionProbs: ArrayLike<number>): void {
  for (let i = 0; i < inclusionProbs.length; i++) {
    const p = inclusionProbs[i];
    if (!(p > 0 && p <= 1)) throw new Error(`inclusionProbs[${i}] must be in (0, 1], got ${p}`);
  }
}

function checkLengths(n: number, entries: Record<string, { length: number } | undefined>): void {
  for (const [name, v] of Object.entries(entries)) if (v !== undefined && v.length !== n) throw new Error(`${name} has ${v.length} entries, expected ${n}`);
}

/** Horvitz-Thompson total: Σ y_i / π_i. */
export function htTotal(values: ArrayLike<number>, inclusionProbs: ArrayLike<number>): number {
  checkLengths(values.length, { inclusionProbs });
  checkFinite('values', values);
  checkProbs(inclusionProbs);
  let total = 0;
  for (let i = 0; i < values.length; i++) total += values[i] / inclusionProbs[i];
  return total;
}

export interface StratifiedDesign {
  inclusionProbs: ArrayLike<number>;
  strata: readonly string[];
  /** N_h: population size of every stratum that appears in `strata`. */
  stratumSizes: Readonly<Record<string, number>>;
}

interface Stratum { name: string; idx: number[]; N: number }

/** Groups units by stratum and checks the design is stratified SRS: π_i = n_h / N_h for every unit. */
function strataOf({ inclusionProbs, strata, stratumSizes }: StratifiedDesign, n: number): Stratum[] {
  checkLengths(n, { inclusionProbs, strata });
  checkProbs(inclusionProbs);
  const by = new Map<string, number[]>();
  strata.forEach((s, i) => (by.get(s) ?? by.set(s, []).get(s)!).push(i));
  return [...by.entries()].map(([name, idx]) => {
    const N = stratumSizes[name];
    if (!(Number.isFinite(N) && N >= idx.length)) throw new Error(`stratum ${name}: N_h = ${N} but ${idx.length} units were sampled`);
    const pi = idx.length / N;
    for (const i of idx) {
      if (Math.abs(inclusionProbs[i] - pi) > 1e-9 * Math.max(1, pi)) {
        throw new Error(`stratum ${name}: inclusionProbs[${i}] = ${inclusionProbs[i]} but n_h / N_h = ${pi} (stratified simple random sampling only)`);
      }
    }
    return { name, idx, N };
  });
}

/** Linearised (Taylor) estimate of a ratio of totals, R̂ = Â / B̂, under stratified SRS without replacement. */
export function stratifiedRatio(options: StratifiedDesign & { num: ArrayLike<number>; den: ArrayLike<number> }): { estimate: number; variance: number; se: number } {
  const { num, den } = options;
  checkLengths(num.length, { den });
  checkFinite('num', num);
  checkFinite('den', den);
  const strata = strataOf(options, num.length);
  let A = 0, B = 0;
  for (const s of strata) for (const i of s.idx) {
    A += (num[i] * s.N) / s.idx.length;
    B += (den[i] * s.N) / s.idx.length;
  }
  if (B === 0) throw new Error('the denominator total is 0 - the ratio is undefined');
  const R = A / B;
  let variance = 0;
  for (const s of strata) {
    const nh = s.idx.length;
    if (nh < 2) throw new Error(`stratum ${s.name} has ${nh} sampled unit(s); variance needs at least 2`);
    const fh = nh / s.N;
    if (fh === 1) continue; // a census stratum contributes no sampling variance
    const z = s.idx.map((i) => (num[i] - R * den[i]) / B);
    const mean = z.reduce((a, b) => a + b, 0) / nh;
    const s2 = z.reduce((acc, v) => acc + (v - mean) ** 2, 0) / (nh - 1);
    variance += (s.N * s.N * (1 - fh) * s2) / nh;
  }
  return { estimate: R, variance, se: Math.sqrt(variance) };
}

/** Deterministic 32-bit PRNG (mulberry32): uniform in [0, 1). */
export function seededRandom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Rao-Wu rescaling bootstrap: per replicate and stratum, draw m_h = n_h - 1 units with replacement;
 * a unit drawn r_i times gets weight w_i · n_h / (n_h - 1) · r_i. Returns replicates × units.
 * Ignores the finite-population correction, so it overstates variance when sampling fractions
 * are large; the linearised estimator is the reference then.
 */
export function stratifiedBootstrap(options: { inclusionProbs: ArrayLike<number>; strata: readonly string[]; replicates?: number; seed: number }): number[][] {
  const { inclusionProbs, strata, replicates = 2000, seed } = options;
  checkLengths(inclusionProbs.length, { strata });
  checkProbs(inclusionProbs);
  if (!Number.isInteger(replicates) || replicates < 1) throw new Error(`replicates must be a positive integer, got ${replicates}`);
  const by = new Map<string, number[]>();
  strata.forEach((s, i) => (by.get(s) ?? by.set(s, []).get(s)!).push(i));
  for (const [name, idx] of by) if (idx.length < 2) throw new Error(`stratum ${name} has ${idx.length} sampled unit(s); the bootstrap needs at least 2`);
  const groups = [...by.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)); // seed-stable order
  const rand = seededRandom(seed);
  const out: number[][] = [];
  for (let b = 0; b < replicates; b++) {
    const w = new Array<number>(inclusionProbs.length).fill(0);
    for (const [, idx] of groups) {
      const nh = idx.length;
      for (let d = 0; d < nh - 1; d++) {
        const i = idx[Math.floor(rand() * nh)];
        w[i] += (1 / inclusionProbs[i]) * (nh / (nh - 1));
      }
    }
    out.push(w);
  }
  return out;
}

/** Kish's effective sample size: (Σ w)² / Σ w². */
export function kishEffectiveN(weights: ArrayLike<number>): number {
  checkFinite('weights', weights);
  let s = 0, s2 = 0;
  for (let i = 0; i < weights.length; i++) {
    if (weights[i] < 0) throw new Error(`weights[${i}] is negative`);
    s += weights[i];
    s2 += weights[i] * weights[i];
  }
  return s2 === 0 ? 0 : (s * s) / s2;
}

/** The smallest value whose cumulative normalised weight is at least q (tied values share their group's cumulative weight). */
export function weightedQuantile(values: ArrayLike<number>, weights: ArrayLike<number>, q: number): number {
  checkLengths(values.length, { weights });
  checkFinite('values', values);
  checkFinite('weights', weights);
  if (!(q >= 0 && q <= 1)) throw new Error(`q must be in [0, 1], got ${q}`);
  const order = Array.from(values, (_, i) => i).sort((a, b) => values[a] - values[b]);
  let total = 0;
  for (let i = 0; i < weights.length; i++) {
    if (weights[i] < 0) throw new Error(`weights[${i}] is negative`);
    total += weights[i];
  }
  if (!(total > 0)) throw new Error('weights must not all be zero');
  let cum = 0;
  for (let j = 0; j < order.length; j++) {
    cum += weights[order[j]];
    // Ties share the cumulative weight of the whole tied group.
    if (j + 1 < order.length && values[order[j + 1]] === values[order[j]]) continue;
    if (cum / total >= q - 1e-12) return values[order[j]];
  }
  return values[order[order.length - 1]];
}

/** Standard normal quantile (Acklam's rational approximation, relative error < 1.2e-9). */
export function normalQuantile(p: number): number {
  if (!(p > 0 && p < 1)) throw new Error(`p must be strictly between 0 and 1, got ${p}`);
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const lo = 0.02425;
  if (p < lo) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - lo) return -normalQuantile(1 - p);
  const q = p - 0.5, r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

export type DesignRiskMethod = 'exact' | 'linearised' | 'bootstrap';

export type DesignRiskResult =
  | { feasible: true; threshold: number; missRateEstimate: number; missRateUpper: number; nEff: number; guarantee: 'design-exact' | 'design-approximate'; method: DesignRiskMethod }
  | { feasible: false; reason: string; nEff: number };

/**
 * A recall threshold (act at score >= t) from a stratified sample: the highest candidate whose
 * miss rate - the weighted share of positives scoring below it - has an upper bound <= α.
 * Candidates are the distinct positive scores, scanned upwards; the scan stops at the first failure.
 * Pass every sampled calibration unit, positive and negative.
 *
 *   exact (default)  per stratum, Clopper-Pearson bounds on the population counts of missed
 *                    positives (upper) and caught positives (lower), Bonferroni over 2·H, combined
 *                    as M_U / (M_U + C_L); census strata use their counts. The bound only rises
 *                    with the threshold, so scanning data-dependent candidates keeps it valid with
 *                    probability 1 - δ. Conservative: a stratum with few sampled units but a large
 *                    N_h can make a target infeasible - which is the truth about such a sample.
 *   linearised       R̂ + z_{1-δ}·se, floored by Clopper-Pearson at the Kish effective number of
 *   bootstrap        positives (or the 1 - δ Rao-Wu quantile). APPROXIMATE, and they UNDER-COVER
 *                    when a heavily weighted stratum yields few or no sampled positives: the
 *                    positives it could hold contribute no variance and no weight. In a simulation
 *                    with 13% of positives in such a stratum, linearised failed in 20% of runs at δ = 5%.
 */
export function designRiskThreshold(options: StratifiedDesign & {
  y: ArrayLike<number>;
  scores: ArrayLike<number>;
  alpha: number;
  delta: number;
  method?: DesignRiskMethod;
  replicates?: number;
  seed?: number;
}): DesignRiskResult {
  const { y, scores, alpha, delta, method = 'exact', replicates = 2000, seed = 0 } = options;
  if (!(alpha > 0 && alpha < 1)) throw new Error(`alpha must be strictly between 0 and 1, got ${alpha}`);
  if (!(delta > 0 && delta < 1)) throw new Error(`delta must be strictly between 0 and 1, got ${delta}`);
  checkLengths(y.length, { scores });
  checkFinite('scores', scores);
  for (let i = 0; i < y.length; i++) if (y[i] !== 0 && y[i] !== 1) throw new Error(`y[${i}] must be 0 or 1`);
  const strata = strataOf(options, y.length);
  let hasNegative = false;
  for (let i = 0; i < y.length; i++) if (y[i] === 0) hasNegative = true;
  if (!hasNegative) throw new Error('designRiskThreshold needs every sampled calibration unit, not only positives: the miss-rate variance is a domain estimate over all of them');
  for (const s of strata) if (s.idx.length < 2) throw new Error(`stratum ${s.name} has ${s.idx.length} sampled unit(s); variance needs at least 2`);

  const w = Array.from(y, (_, i) => 1 / options.inclusionProbs[i]);
  const positives = Array.from(y, (_, i) => i).filter((i) => y[i] === 1).sort((a, b) => scores[a] - scores[b]);
  const nEff = kishEffectiveN(positives.map((i) => w[i]));
  if (positives.length === 0) return { feasible: false, reason: 'no calibration positives', nEff: 0 };
  const z = normalQuantile(1 - delta);
  const nFloor = Math.floor(nEff);

  // Per stratum: positives p_h and misses m_h so far; within a stratum every unit has weight N_h/n_h.
  const hOf = new Map<number, number>();
  strata.forEach((s, h) => s.idx.forEach((i) => hOf.set(i, h)));
  const pos = strata.map((s) => s.idx.filter((i) => y[i] === 1).length);
  const miss = strata.map(() => 0);
  const wh = strata.map((s) => s.N / s.idx.length);
  const B = strata.reduce((acc, _, h) => acc + wh[h] * pos[h], 0);

  const reps = method === 'bootstrap' ? stratifiedBootstrap({ inclusionProbs: options.inclusionProbs, strata: options.strata, replicates, seed }) : null;
  const repDen = reps?.map((r) => positives.reduce((acc, i) => acc + r[i], 0));
  const repNum = reps?.map(() => 0);

  // Exact: per-stratum bounds at confidence 1 - δ/(2H), memoised by count (one stratum changes per step).
  const conf = 1 - delta / (2 * strata.length);
  const cpUpper = strata.map(() => new Map<number, number>());
  const cpLower = strata.map(() => new Map<number, number>());
  const memo = (m: Map<number, number>, k: number, f: () => number) => m.get(k) ?? m.set(k, f()).get(k)!;
  const exactUpper = (): number => {
    let MU = 0, CL = 0;
    strata.forEach((s, h) => {
      const nh = s.idx.length, caught = pos[h] - miss[h];
      if (nh === s.N) {
        MU += miss[h];
        CL += caught;
        return;
      }
      MU += s.N * memo(cpUpper[h], miss[h], () => clopperPearsonUpper(miss[h], nh, conf));
      CL += s.N * memo(cpLower[h], caught, () => clopperPearsonLower(caught, nh, conf));
    });
    return MU + CL === 0 ? 1 : MU / (MU + CL);
  };

  const bound = (): { R: number; upper: number } => {
    let A = 0;
    strata.forEach((_, h) => (A += wh[h] * miss[h]));
    const R = A / B;
    if (method === 'exact') return { R, upper: exactUpper() };
    let analytic: number;
    if (reps) {
      const rs = reps.map((_, b) => (repDen![b] > 0 ? repNum![b] / repDen![b] : 1)); // no positives drawn: assume the worst
      analytic = weightedQuantile(rs, rs.map(() => 1), 1 - delta);
    } else {
      let variance = 0;
      strata.forEach((s, h) => {
        const nh = s.idx.length;
        const fh = nh / s.N;
        if (fh === 1) return;
        // z_i: (1 - R)/B for a missed positive, -R/B for a caught one, 0 for a negative.
        const sum = (miss[h] * (1 - R) - (pos[h] - miss[h]) * R) / B;
        const sumSq = (miss[h] * (1 - R) ** 2 + (pos[h] - miss[h]) * R * R) / (B * B);
        const s2 = (sumSq - (sum * sum) / nh) / (nh - 1);
        variance += (s.N * s.N * (1 - fh) * Math.max(0, s2)) / nh;
      });
      analytic = R + z * Math.sqrt(variance);
    }
    const floor = nFloor >= 1 ? clopperPearsonUpper(Math.min(nFloor, Math.ceil(R * nEff - 1e-9)), nFloor, 1 - delta) : 1;
    return { R, upper: Math.max(analytic, floor) };
  };

  let best: { t: number; R: number; upper: number } | null = null;
  let j = 0;
  while (j < positives.length) {
    const t = scores[positives[j]];
    const b = bound(); // misses so far are exactly the positives scoring below t
    if (b.upper > alpha) {
      if (!best) return { feasible: false, reason: `the miss-rate bound is ${b.upper.toFixed(4)} > α = ${alpha} even at the lowest positive score (Kish effective positives ${nEff.toFixed(1)})`, nEff };
      break;
    }
    best = { t, R: b.R, upper: b.upper };
    // Advance past every positive tied at t: they become misses for the next candidate.
    while (j < positives.length && scores[positives[j]] === t) {
      const i = positives[j];
      miss[hOf.get(i)!]++;
      if (reps) reps.forEach((r, k) => (repNum![k] += r[i]));
      j++;
    }
  }
  return { feasible: true, threshold: best!.t, missRateEstimate: best!.R, missRateUpper: best!.upper, nEff, guarantee: method === 'exact' ? 'design-exact' : 'design-approximate', method };
}
