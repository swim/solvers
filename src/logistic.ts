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
 * (hundreds of features), impractical beyond a few thousand features.
 */
import { CholeskyDecomposition, Matrix } from 'ml-matrix';

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

  const theta = new Float64Array(m);
  const z = new Float64Array(n);

  const objectiveAt = (params: Float64Array): number => {
    let loss = 0;
    for (let i = 0; i < n; i++) {
      let zi = params[d];
      const xi = X[i];
      for (let j = 0; j < d; j++) zi += params[j] * xi[j];
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
    const grad = new Float64Array(m);
    const hess = new Float64Array(m * m);
    for (let i = 0; i < n; i++) {
      const xi = X[i];
      let zi = theta[d];
      for (let j = 0; j < d; j++) zi += theta[j] * xi[j];
      z[i] = zi;
      const p = sigmoid(zi);
      const r = C * s[i] * (p - y[i]);
      const h = C * s[i] * p * (1 - p);
      for (let j = 0; j < d; j++) {
        grad[j] += r * xi[j];
        const hj = h * xi[j];
        const row = j * m;
        for (let k = j; k < d; k++) hess[row + k] += hj * xi[k];
        hess[row + d] += hj; // x_j × intercept
      }
      grad[d] += r;
      hess[d * m + d] += h;
    }
    for (let j = 0; j < d; j++) {
      grad[j] += theta[j];
      hess[j * m + j] += 1;
    }
    for (let j = 0; j < m; j++) for (let k = 0; k < j; k++) hess[j * m + k] = hess[k * m + j];

    const H = new Matrix(Array.from({ length: m }, (_, j) => hess.subarray(j * m, (j + 1) * m)));
    const chol = new CholeskyDecomposition(H);
    if (!chol.isPositiveDefinite()) throw new Error('Hessian is not positive definite - check the data for NaN/constant columns');
    const step = chol.solve(Matrix.columnVector(Array.from(grad))).to1DArray();

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
