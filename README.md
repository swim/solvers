# @liquidau/solvers

Exact, dependency-free statistics for binary classifiers in TypeScript, checked against
scikit-learn, MAPIE, crepes, samplics and scipy.

```
scores ──► fit ──────────► calibrate ──────► threshold ───────► check
           fitLogistic     fitPlatt          conformal*         coxTest
                           fitIsotonic       designRisk*        exceedanceTest
                                             designPrecision*   wilson, clopperPearson*
```

Used by [`@liquidau/rule-miner`](https://www.npmjs.com/package/@liquidau/rule-miner) (rule bounds)
and [`@liquidau/embedding-classifier`](https://www.npmjs.com/package/@liquidau/embedding-classifier)
(training, thresholds, gates).

## Install

```bash
npm install @liquidau/solvers
```

## Example

```ts
import { decisionFunction, fitLogistic, fitPlatt, predictPlatt, prevalenceWeights, seededRandom } from '@liquidau/solvers';

// Toy data: two features, positives shifted up. Use your embeddings and labels instead.
const rand = seededRandom(1);
const make = (n: number) => Array.from({ length: n }, () => { const y = rand() < 0.3 ? 1 : 0; return { x: [rand() + y, rand() + y], y }; });
const train = make(400), cal = make(200);

const model = fitLogistic(train.map((d) => d.x), train.map((d) => d.y), { C: 1, classWeight: 'balanced' });
const logits = cal.map((d) => decisionFunction(model, d.x));
// Calibrate to the prevalence expected in production (1%), not the training mix.
const platt = fitPlatt(logits, cal.map((d) => d.y), prevalenceWeights(cal.map((d) => d.y), 0.01));
console.log(predictPlatt(platt, decisionFunction(model, [1.2, 1.1]))); // calibrated probability
```

## What's in it

| Area | Exports |
|---|---|
| Logistic regression | `fitLogistic`, `decisionFunction`, `sigmoid` |
| Calibration | `fitPlatt`, `fitIsotonic`, `prevalenceWeights`, `coxTest`, `ece` |
| Bounds | `wilson`, `clopperPearsonUpper`, `clopperPearsonLower`, `binomialCdf` |
| Conformal thresholds | `conformalLowerThreshold`, `conformalUpperThreshold`, `minimumSamples` |
| Stratified samples | `htTotal`, `stratifiedRatio`, `stratifiedBootstrap`, `kishEffectiveN` |
| Design thresholds | `designRiskThreshold`, `designPrecisionThreshold` |
| Monitoring, agreement | `exceedanceTest`, `cohenKappa` |

Every export is listed in [docs/API.md](docs/API.md).

## Guarantees and limits

- Logistic coefficients match scikit-learn to 1e-6, probabilities to 1e-8.
- Conformal thresholds match MAPIE and crepes exactly.
- Binary only, L2 only, dense features.
- About 6 s to fit 3,000 × 768 on a laptop; cost grows with features squared.
- **Design thresholds:** `linearised` bounds can under-cover with thin strata, so `exact` is the default.

## More

- [docs/API.md](docs/API.md): every export, and why the solvers are exact.
- [docs/VERIFICATION.md](docs/VERIFICATION.md): golden fixtures, tolerances, simulations, timing.
- [docs/SKLEARN-DIFFERENCES.md](docs/SKLEARN-DIFFERENCES.md): where results differ from scikit-learn, and why.

## Develop

```bash
npm install
npm test        # runs the .ts tests through tsx, so Node 20 works
npm run build   # dist/ (ESM + .d.ts)
```

## License

MIT. See [LICENSE](LICENSE). The isotonic regression in `src/isotonic.ts` is ported from
scikit-learn and keeps its BSD-3-Clause notice in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
