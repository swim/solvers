/**
 * Weighted isotonic regression (non-decreasing), ported step for step from scikit-learn's
 * IsotonicRegression(increasing=True, out_of_bounds="clip") so the fitted thresholds match:
 *
 *   0. drop samples with zero weight                - sklearn _build_y (sample_weight > 0)
 *   1. sort by (x, then y)                          - np.lexsort((y, X))
 *   2. merge equal x: weighted-mean y, summed weight - sklearn _make_unique (eps = float64 resolution)
 *   3. weighted pool-adjacent-violators             - sklearn _inplace_contiguous_isotonic_regression
 *   4. clip to [yMin, yMax]
 *   5. drop interior points equal to both neighbours (scikit-learn's trim_duplicates)
 *
 * predict() is linear interpolation between thresholds, clipped at both ends (np.interp).
 *
 * Derived from scikit-learn, Copyright (c) 2007-2026 The scikit-learn developers, BSD-3-Clause.
 * See THIRD_PARTY_NOTICES for the full license text.
 *
 * Unlike scikit-learn, which silently drops negative weights along with zero ones, negative or
 * non-finite weights throw here, as do non-finite x or y.
 */

export interface IsotonicModel {
  x: number[];
  y: number[];
}

const FLOAT64_RESOLUTION = 1e-15; // np.finfo(np.float64).resolution

export function fitIsotonic(
  xIn: ArrayLike<number>,
  yIn: ArrayLike<number>,
  sampleWeight?: ArrayLike<number>,
  bounds: { yMin?: number; yMax?: number } = {},
): IsotonicModel {
  const n = xIn.length;
  if (n === 0 || n !== yIn.length) throw new Error('x and y must be non-empty and the same length');
  if (sampleWeight && sampleWeight.length !== n) throw new Error(`sampleWeight has length ${sampleWeight.length}, expected ${n}`);

  // 0. validate, and drop zero-weight samples (they would otherwise give 0/0 = NaN means)
  const order: number[] = [];
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(xIn[i]) || !Number.isFinite(yIn[i])) throw new Error('x and y must be finite');
    const wi = sampleWeight ? sampleWeight[i] : 1;
    if (!Number.isFinite(wi) || wi < 0) throw new Error('sample weights must be finite and non-negative');
    if (wi > 0) order.push(i);
  }
  if (order.length === 0) throw new Error('sample weights must not all be zero');

  // 1. sort
  order.sort((a, b) => xIn[a] - xIn[b] || yIn[a] - yIn[b]);

  // 2. unique x
  const ux: number[] = [];
  const uy: number[] = [];
  const uw: number[] = [];
  let currentX = xIn[order[0]];
  let sumWY = 0;
  let sumW = 0;
  for (const i of order) {
    const xi = xIn[i];
    const wi = sampleWeight ? sampleWeight[i] : 1;
    if (xi - currentX >= FLOAT64_RESOLUTION) {
      ux.push(currentX);
      uy.push(sumWY / sumW);
      uw.push(sumW);
      currentX = xi;
      sumWY = 0;
      sumW = 0;
    }
    sumWY += yIn[i] * wi;
    sumW += wi;
  }
  ux.push(currentX);
  uy.push(sumWY / sumW);
  uw.push(sumW);

  // 3. PAVA, in place on uy/uw
  const m = uy.length;
  const target = Array.from({ length: m }, (_, i) => i);
  let i = 0;
  while (i < m) {
    let k = target[i] + 1;
    if (k === m) break;
    if (uy[i] < uy[k]) {
      i = k;
      continue;
    }
    let sumWy = uw[i] * uy[i];
    let sumWeights = uw[i];
    for (;;) {
      const prevY = uy[k];
      sumWy += uw[k] * uy[k];
      sumWeights += uw[k];
      k = target[k] + 1;
      if (k === m || prevY < uy[k]) {
        uy[i] = sumWy / sumWeights;
        uw[i] = sumWeights;
        target[i] = k - 1;
        target[k - 1] = i;
        if (i > 0) i = target[i - 1];
        break;
      }
    }
  }
  for (i = 0; i < m; ) {
    const k = target[i] + 1;
    for (let j = i + 1; j < k; j++) uy[j] = uy[i];
    i = k;
  }

  // 4. clip
  for (let j = 0; j < m; j++) {
    if (bounds.yMin !== undefined && uy[j] < bounds.yMin) uy[j] = bounds.yMin;
    if (bounds.yMax !== undefined && uy[j] > bounds.yMax) uy[j] = bounds.yMax;
  }

  // 5. trim redundant interior points
  const x: number[] = [];
  const y: number[] = [];
  for (let j = 0; j < m; j++) {
    const keep = j === 0 || j === m - 1 || uy[j - 1] !== uy[j] || uy[j] !== uy[j + 1];
    if (keep) {
      x.push(ux[j]);
      y.push(uy[j]);
    }
  }
  return { x, y };
}

export function predictIsotonic(model: IsotonicModel, value: number): number {
  const { x, y } = model;
  if (Number.isNaN(value)) return NaN;
  if (value <= x[0]) return y[0];
  if (value >= x[x.length - 1]) return y[y.length - 1];
  let lo = 0;
  let hi = x.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (x[mid] <= value) lo = mid;
    else hi = mid;
  }
  return x[hi] === x[lo] ? y[lo] : y[lo] + ((y[hi] - y[lo]) * (value - x[lo])) / (x[hi] - x[lo]);
}
