/**
 * Distribution-free thresholds from order statistics (split conformal, one class at a time).
 *
 * With n exchangeable calibration scores, a threshold at the (k+1)-th smallest leaves a share
 * U_(k+1) of future scores below it - a uniform order statistic - so
 *
 *   PAC        P(share below > α) = P(Bin(n, α) <= k)    k = the largest with this <= δ
 *   expected   E[share below] = (k + 1) / (n + 1)          k = ⌊α(n + 1)⌋ - 1
 *
 * and symmetrically from the top. Learn Then Test with fixed-sequence testing over thresholds
 * (exact binomial p-values, strictest first) stops at exactly this rank, so these are that
 * procedure in closed form. Verified against MAPIE's BinaryClassificationController and crepes'
 * class-conditional p-values (test/fixtures/conformal-golden.json).
 *
 * The guarantees hold at any sample size and survive any non-decreasing transform of the scores
 * (ties only make them more conservative). Scores must be exchangeable with the future ones they
 * describe; with paraphrases or near-duplicates, pass one score per group.
 */
import { binomialCdf } from './metrics.ts';

function checkRate(name: string, v: number): void {
  if (!(v > 0 && v < 1)) throw new Error(`${name} must be strictly between 0 and 1, got ${v}`);
}

/**
 * How many of n calibration scores may fall on the wrong side of the threshold: with δ, the largest
 * k with P(Bin(n, α) <= k) <= δ; without, ⌊α(n + 1)⌋ - 1. -1 means no threshold gives the guarantee.
 */
export function conformalRank(n: number, alpha: number, delta?: number): number {
  if (!Number.isInteger(n) || n < 0) throw new Error(`n must be a non-negative integer, got ${n}`);
  checkRate('alpha', alpha);
  if (delta === undefined) return Math.floor(alpha * (n + 1)) - 1;
  checkRate('delta', delta);
  let k = -1;
  while (k + 1 < n && binomialCdf(k + 1, n, alpha) <= delta) k++;
  return k;
}

/** The fewest calibration scores for any guarantee at α (and δ): zero allowed on the wrong side. */
export function minimumSamples(alpha: number, delta?: number): number {
  checkRate('alpha', alpha);
  if (delta !== undefined) checkRate('delta', delta);
  // α(n + 1) >= 1, or (1 - α)^n <= δ. Start just below the closed form, step up exactly.
  const estimate = delta === undefined ? 1 / alpha - 1 : Math.log(delta) / Math.log1p(-alpha);
  let n = Math.max(0, Math.floor(estimate) - 2);
  while (conformalRank(n, alpha, delta) < 0) n++;
  return n;
}

/**
 * The highest threshold t (act at score >= t) with at most a share α of future scores below it -
 * e.g. a recall threshold from positives' scores. The (k+1)-th smallest score, or null.
 */
export function conformalLowerThreshold(scores: ArrayLike<number>, alpha: number, delta?: number): number | null {
  const sorted = Float64Array.from(scores).sort();
  const k = conformalRank(sorted.length, alpha, delta);
  return k < 0 ? null : sorted[k];
}

/**
 * The lowest threshold t (act at score >= t) with at most a share α of future scores at or above
 * it - e.g. a false-alarm threshold from negatives' scores. Just above the (k+1)-th largest, or null.
 */
export function conformalUpperThreshold(scores: ArrayLike<number>, alpha: number, delta?: number): number | null {
  const sorted = Float64Array.from(scores).sort().reverse();
  const k = conformalRank(sorted.length, alpha, delta);
  return k < 0 ? null : nextUp(sorted[k]);
}

/** Smallest double strictly greater than v, so `score >= nextUp(v)` excludes v itself. */
export function nextUp(v: number): number {
  if (Number.isNaN(v) || v === Infinity) return v;
  if (v === 0) return Number.MIN_VALUE;
  const buf = new DataView(new ArrayBuffer(8));
  buf.setFloat64(0, v);
  const bits = buf.getBigUint64(0);
  buf.setBigUint64(0, v > 0 ? bits + 1n : bits - 1n);
  return buf.getFloat64(0);
}
