/**
 * Drift monitoring for a deployed score: is the live share of scores at or above a threshold larger
 * than a bound? An exact one-sided binomial test - aimed at the tail, where a classifier's threshold
 * sits. (A two-sample Kolmogorov-Smirnov test on the whole score distribution was evaluated and
 * removed: in simulation it missed tail drift that this test caught, and caught nothing it missed.)
 *
 * Compare like with like (same weekday, same hours), since traffic is rarely exchangeable over time
 * even without drift. A change in P(label | text) can't be seen this way: that needs fresh labelled samples.
 */
import { binomialCdf, clopperPearsonLower } from './metrics.ts';

/**
 * Exact one-sided test of H0: rate <= bound, from `count` exceedances in `n` live scores. Alerts
 * when the p-value P(Bin(n, bound) >= count) is at most alpha; `lower` is the exact 1 - alpha lower
 * bound on the live rate (an alert means lower > bound).
 */
export function exceedanceTest(count: number, n: number, bound: number, alpha = 0.05): { rate: number; lower: number; pValue: number; alert: boolean } {
  if (!(Number.isInteger(count) && Number.isInteger(n) && count >= 0 && count <= n && n > 0)) throw new Error(`need integers 0 <= count <= n, n > 0 (got ${count}, ${n})`);
  if (!(bound >= 0 && bound <= 1)) throw new Error(`bound must be in [0, 1], got ${bound}`);
  if (!(alpha > 0 && alpha < 1)) throw new Error(`alpha must be strictly between 0 and 1, got ${alpha}`);
  const pValue = count === 0 ? 1 : 1 - binomialCdf(count - 1, n, bound);
  return { rate: count / n, lower: clopperPearsonLower(count, n, 1 - alpha), pValue, alert: pValue <= alpha };
}
