/**
 * Binary logistic regression with L2 regularisation and per-sample weights, solved exactly by
 * damped Newton's method (IRLS). Minimises the same objective as scikit-learn's
 * LogisticRegression(penalty="l2", fit_intercept=True):
 *
 *     C * Σ sᵢ · logloss(yᵢ, σ(w·xᵢ + b))  +  ½ ‖w‖²        (intercept not penalised)
 *
 * The objective is strictly convex, so there is exactly one optimum - any correct solver reaches
 * the same coefficients, which is what makes this verifiable against scikit-learn fixtures.
 *
 * Each Newton step solves a (d+1)×(d+1) system by Cholesky: fast for embedding-sized problems
 * (hundreds of features), impractical beyond a few thousand features. X is copied once into one
 * contiguous Float64Array and the factorisation runs in place, so an iteration allocates nothing.
 */

/**
 * Solves A x = b for symmetric positive-definite A (m × m, row-major), in place: `L` receives the
 * Cholesky factor (lower triangle) and `x` the solution. Returns false if A isn't positive definite.
 */
function choleskySolve(A: Float64Array, b: Float64Array, m: number, L: Float64Array, x: Float64Array): boolean {
  for (let j = 0; j < m; j++) {
    const rj = j * m;
    let diag = A[rj + j];
    for (let k = 0; k < j; k++) diag -= L[rj + k] * L[rj + k];
    if (!(diag > 0)) return false;
    const ljj = Math.sqrt(diag);
    L[rj + j] = ljj;
    for (let i = j + 1; i < m; i++) {
      const ri = i * m;
      let v = A[ri + j];
      for (let k = 0; k < j; k++) v -= L[ri + k] * L[rj + k];
      L[ri + j] = v / ljj;
    }
  }
  // Forward (L z = b), then back substitution (Lᵀ x = z).
  for (let i = 0; i < m; i++) {
    const ri = i * m;
    let v = b[i];
    for (let k = 0; k < i; k++) v -= L[ri + k] * x[k];
    x[i] = v / L[ri + i];
  }
  for (let i = m - 1; i >= 0; i--) {
    let v = x[i];
    for (let k = i + 1; k < m; k++) v -= L[k * m + i] * x[k];
    x[i] = v / L[i * m + i];
  }
  return true;
}

export interface LogisticOptions {
  /** Inverse regularisation strength, as scikit-learn's C. */
  C?: number;
  /** Per-sample weights (default all 1). */
  sampleWeight?: ArrayLike<number>;
  /** "balanced" multiplies weights by n / (2 · count(class)), as scikit-learn's class_weight. */
  classWeight?: 'balanced' | null;
  /** Stop when half the squared Newton decrement falls below this. */
  tol?: number;
  maxIter?: number;
}

export interface LogisticModel {
  coef: number[];
  intercept: number;
  iterations: number;
  objective: number;
  /**
   * True when the Newton-decrement stopping test was met. False when the fit hit `maxIter`, or
   * stopped because the line search could no longer decrease the objective at float precision
   * before the decrement test passed.
   */
  converged: boolean;
}

export function sigmoid(z: number): number {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

/** log(1 + eᶻ) without overflow. */
function softplus(z: number): number {
  return z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z));
}

export function decisionFunction(model: Pick<LogisticModel, 'coef' | 'intercept'>, x: ArrayLike<number>): number {
  if (x.length !== model.coef.length) throw new Error(`x has ${x.length} features, the model expects ${model.coef.length}`);
  let z = model.intercept;
  for (let j = 0; j < model.coef.length; j++) z += model.coef[j] * x[j];
  return z;
}

export function effectiveWeights(y: ArrayLike<number>, options: LogisticOptions): Float64Array {
  const n = y.length;
  const { sampleWeight } = options;
  if (sampleWeight && sampleWeight.length !== n) throw new Error(`sampleWeight has length ${sampleWeight.length}, expected ${n}`);
  const w = new Float64Array(n);
  let total = 0;
  for (let i = 0; i < n; i++) {
    if (y[i] !== 0 && y[i] !== 1) throw new Error('y must be 0/1');
    const wi = sampleWeight ? sampleWeight[i] : 1;
    if (!Number.isFinite(wi) || wi < 0) throw new Error('sample weights must be finite and non-negative');
    w[i] = wi;
    total += wi;
  }
  if (!(total > 0)) throw new Error('sample weights must not all be zero');
  if (options.classWeight === 'balanced') {
    // Class totals are SAMPLE-WEIGHTED, as in current scikit-learn (compute_class_weight receives
    // sample_weight): weight_c = Σw / (2 · Σ_{i in c} wᵢ). With unit weights this reduces to the
    // familiar n / (2 · count_c). Caught by the golden test combining both options.
    let positives = 0;
    for (let i = 0; i < n; i++) if (y[i] === 1) positives += w[i];
    const negatives = total - positives;
    if (positives === 0 || negatives === 0) throw new Error('classWeight "balanced" needs both classes');
    for (let i = 0; i < n; i++) w[i] *= y[i] === 1 ? total / (2 * positives) : total / (2 * negatives);
  }
  return w;
}

export function fitLogistic(X: ArrayLike<number>[], y: ArrayLike<number>, options: LogisticOptions = {}): LogisticModel {
  const { C = 1, tol = 1e-14, maxIter = 100 } = options;
  const n = X.length;
  if (n === 0 || n !== y.length) throw new Error('X and y must be non-empty and the same length');
  const d = X[0].length;
  const m = d + 1; // last parameter is the intercept
  for (let i = 0; i < n; i++) {
    if (y[i] !== 0 && y[i] !== 1) throw new Error('y must be 0/1');
    if (X[i].length !== d) throw new Error('every row of X must have the same length');
  }
  const s = effectiveWeights(y, options);
  // One contiguous copy of X: typed, row-major, cache-friendly.
  const Xf = new Float64Array(n * d);
  for (let i = 0; i < n; i++) {
    const xi = X[i], off = i * d;
    for (let j = 0; j < d; j++) Xf[off + j] = xi[j];
  }

  const theta = new Float64Array(m);
  const z = new Float64Array(n);
  const grad = new Float64Array(m);
  const hess = new Float64Array(m * m);
  const factor = new Float64Array(m * m);
  const step = new Float64Array(m);
  const rr = new Float64Array(n);
  const hh = new Float64Array(n);

  const objectiveAt = (params: Float64Array): number => {
    let loss = 0;
    for (let i = 0; i < n; i++) {
      let zi = params[d];
      const off = i * d;
      for (let j = 0; j < d; j++) zi += params[j] * Xf[off + j];
      loss += s[i] * (softplus(zi) - y[i] * zi);
    }
    let penalty = 0;
    for (let j = 0; j < d; j++) penalty += params[j] * params[j];
    return C * loss + 0.5 * penalty;
  };

  let objective = objectiveAt(theta);
  let iterations = 0;
  let converged = false;
  for (; iterations < maxIter; iterations++) {
    // Gradient and Hessian of the objective at theta (upper triangle accumulated, then mirrored).
    grad.fill(0);
    hess.fill(0);
    // Per-sample residual r_i and curvature h_i.
    for (let i = 0; i < n; i++) {
      const off = i * d;
      let zi = theta[d];
      for (let j = 0; j < d; j++) zi += theta[j] * Xf[off + j];
      z[i] = zi;
      const p = sigmoid(zi);
      rr[i] = C * s[i] * (p - y[i]);
      hh[i] = C * s[i] * p * (1 - p);
      for (let j = 0; j < d; j++) grad[j] += rr[i] * Xf[off + j];
      grad[d] += rr[i];
    }
    // Hessian upper triangle, Σ h_i x_i x_iᵀ: four samples per pass over the triangle, so it is
    // read and written a quarter as often (the loop is bound by that memory traffic).
    let i = 0;
    for (; i + 4 <= n; i += 4) {
      const o0 = i * d, o1 = o0 + d, o2 = o1 + d, o3 = o2 + d;
      const h0 = hh[i], h1 = hh[i + 1], h2 = hh[i + 2], h3 = hh[i + 3];
      for (let j = 0; j < d; j++) {
        const a0 = h0 * Xf[o0 + j], a1 = h1 * Xf[o1 + j], a2 = h2 * Xf[o2 + j], a3 = h3 * Xf[o3 + j];
        const row = j * m;
        for (let k = j; k < d; k++) hess[row + k] += a0 * Xf[o0 + k] + a1 * Xf[o1 + k] + a2 * Xf[o2 + k] + a3 * Xf[o3 + k];
        hess[row + d] += a0 + a1 + a2 + a3; // x_j × intercept
      }
      hess[d * m + d] += h0 + h1 + h2 + h3;
    }
    for (; i < n; i++) {
      const off = i * d, h = hh[i];
      for (let j = 0; j < d; j++) {
        const hj = h * Xf[off + j];
        const row = j * m;
        for (let k = j; k < d; k++) hess[row + k] += hj * Xf[off + k];
        hess[row + d] += hj;
      }
      hess[d * m + d] += h;
    }
    for (let j = 0; j < d; j++) {
      grad[j] += theta[j];
      hess[j * m + j] += 1;
    }
    for (let j = 0; j < m; j++) for (let k = 0; k < j; k++) hess[j * m + k] = hess[k * m + j];

    if (!choleskySolve(hess, grad, m, factor, step)) throw new Error('Hessian is not positive definite - check the data for NaN/constant columns');

    // Newton decrement: gᵀH⁻¹g. Half of it estimates the remaining objective gap.
    let decrement = 0;
    for (let j = 0; j < m; j++) decrement += grad[j] * step[j];
    if (decrement / 2 <= tol * Math.max(1, Math.abs(objective))) {
      converged = true;
      break;
    }

    // Backtracking (Armijo) line search - full Newton steps almost always succeed.
    let t = 1;
    const candidate = new Float64Array(m);
    let next = Infinity;
    for (let tries = 0; tries < 50; tries++, t /= 2) {
      for (let j = 0; j < m; j++) candidate[j] = theta[j] - t * step[j];
      next = objectiveAt(candidate);
      if (next <= objective - 1e-4 * t * decrement) break;
    }
    if (!(next < objective)) break; // no further decrease possible at float precision
    theta.set(candidate);
    objective = next;
  }

  return { coef: Array.from(theta.subarray(0, d)), intercept: theta[d], iterations, objective, converged };
}
