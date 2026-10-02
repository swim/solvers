/** Evaluation and calibration metrics - dependency-free; ECE binning matches numpy. */

export function wilson(k: number, n: number, z = 1.96): [number, number] {
  if (n === 0) return [0, 1];
  const p = k / n;
  const den = 1 + (z * z) / n;
  const centre = (p + (z * z) / (2 * n)) / den;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / den;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

/**
 * Reweights an enriched split so positives carry `targetPrevalence` of the total weight -
 * calibrating on the enriched mix directly would make production probabilities too high.
 */
export function prevalenceWeights(y: ArrayLike<number>, targetPrevalence: number): number[] {
  if (!(targetPrevalence > 0 && targetPrevalence < 1)) throw new Error('targetPrevalence must be strictly between 0 and 1');
  if (y.length === 0) throw new Error('y must be non-empty');
  let positives = 0;
  for (let i = 0; i < y.length; i++) {
    if (y[i] !== 0 && y[i] !== 1) throw new Error('y must be 0/1');
    positives += y[i];
  }
  const observed = positives / y.length;
  if (observed === 0 || observed === 1) throw new Error("split has only one class - can't reweight to a target prevalence");
  return Array.from(y, (v) => (v === 1 ? targetPrevalence / observed : (1 - targetPrevalence) / (1 - observed)));
}

export interface ReliabilityRow {
  bin: string;
  n: number;
  mean_predicted: number;
  observed_rate: number;
}

const round4 = (v: number) => Math.round(v * 1e4) / 1e4;

/**
 * Weighted expected calibration error over equal-width bins. Bin edges use numpy linspace's
 * arithmetic (i × step, last edge exactly 1) and np.digitize's rule: bins are closed on the LEFT,
 * so a probability exactly on an interior edge goes to the upper bin (0.5 -> "0.5-0.6").
 * scikit-learn's calibration_curve closes bins on the right (0.5 -> "0.4-0.5"), so per-bin
 * results can differ from it for probabilities that sit exactly on an edge.
 */
export function ece(p: number[], y: number[], w: number[], bins = 10): { ece: number; reliability: ReliabilityRow[] } {
  if (!Number.isInteger(bins) || bins < 1) throw new Error('bins must be a positive integer');
  if (p.length !== y.length || p.length !== w.length) throw new Error('p, y and w must be the same length');
  const step = 1 / bins;
  const edges = Array.from({ length: bins + 1 }, (_, i) => (i === bins ? 1 : i * step));
  const interior = edges.slice(1, -1);
  // Enough decimals to tell the edges apart: 1 for 10 bins, 2 for 20, up to 4.
  let decimals = 1;
  while (decimals < 4 && edges.some((e) => Math.abs(e * 10 ** decimals - Math.round(e * 10 ** decimals)) > 1e-9)) decimals++;

  const wb = new Float64Array(bins);
  const wp = new Float64Array(bins);
  const wy = new Float64Array(bins);
  const count = new Int32Array(bins);
  let total = 0;
  for (let i = 0; i < p.length; i++) {
    // np.digitize(p, interior): number of interior edges <= p (binary search; edges ascend).
    let lo = 0;
    let hi = interior.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (interior[mid] <= p[i]) lo = mid + 1;
      else hi = mid;
    }
    const b = lo;
    wb[b] += w[i];
    wp[b] += w[i] * p[i];
    wy[b] += w[i] * y[i];
    count[b]++;
    total += w[i];
  }
  if (!(total > 0)) throw new Error('total weight must be positive');

  const rows: ReliabilityRow[] = [];
  let err = 0;
  for (let b = 0; b < bins; b++) {
    if (count[b] === 0) continue;
    const conf = wp[b] / wb[b];
    const acc = wy[b] / wb[b];
    err += (wb[b] / total) * Math.abs(acc - conf);
    rows.push({
      bin: `${edges[b].toFixed(decimals)}-${edges[b + 1].toFixed(decimals)}`,
      n: count[b],
      mean_predicted: round4(conf),
      observed_rate: round4(acc),
    });
  }
  return { ece: err, reliability: rows };
}

/**
 * Cohen's kappa for two raters. When chance agreement is 1 (both raters used one identical label
 * throughout) this returns 1, where scikit-learn's cohen_kappa_score returns NaN.
 */
export function cohenKappa(a: string[], b: string[]): number {
  if (a.length !== b.length) throw new Error('a and b must be the same length');
  if (a.length === 0) throw new Error('a and b must be non-empty');
  const n = a.length;
  let agree = 0;
  const ca = new Map<string, number>();
  const cb = new Map<string, number>();
  for (let i = 0; i < n; i++) {
    if (a[i] === b[i]) agree++;
    ca.set(a[i], (ca.get(a[i]) ?? 0) + 1);
    cb.set(b[i], (cb.get(b[i]) ?? 0) + 1);
  }
  const observed = agree / n;
  let expected = 0;
  for (const [k, v] of ca) expected += v * (cb.get(k) ?? 0);
  expected /= n * n;
  return expected === 1 ? 1 : (observed - expected) / (1 - expected);
}

/** P(X <= k) for X ~ Binomial(n, p), summed in log space (exact; fine for n up to ~1e6). */
export function binomialCdf(k: number, n: number, p: number): number {
  if (k < 0) return 0;
  if (k >= n || p <= 0) return 1;
  if (p >= 1) return 0;
  let logTerm = n * Math.log1p(-p); // log P(X = 0)
  let sum = Math.exp(logTerm);
  const ratio = Math.log(p) - Math.log1p(-p);
  for (let i = 1; i <= k; i++) {
    logTerm += Math.log(n - i + 1) - Math.log(i) + ratio;
    sum += Math.exp(logTerm);
  }
  return Math.min(1, sum);
}

/**
 * Exact one-sided Clopper-Pearson upper bound: the largest rate p with P(X <= k | n, p) >= 1 - confidence.
 * With k = 0 this is 1 - (1 - confidence)^(1/n) - the "rule of three" (~3/n at 95%) made exact.
 */
export function clopperPearsonUpper(k: number, n: number, confidence = 0.95): number {
  if (n === 0 || k >= n) return 1;
  const alpha = 1 - confidence;
  let lo = k / n;
  let hi = 1;
  for (let i = 0; i < 100; i++) {
    const mid = (lo + hi) / 2;
    if (binomialCdf(k, n, mid) > alpha) lo = mid;
    else hi = mid;
  }
  return hi;
}
