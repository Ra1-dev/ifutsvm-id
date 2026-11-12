# Research evidence notes

The compact release includes these summaries. To recompute them from predictions,
first run the complete main benchmark, both research experiments and the theory
check described in README, putting the research outputs in the main output
directory's `research/` subdirectory. Then run
`python -m experiments.research_report --output <that-directory>`.

IFUTSVM-ID (corrected) does not beat the simple imbalance-aware baselines under the shared tuning protocol on these datasets. Equal-weight hold-out balanced accuracy over 12 datasets: IFUTSVM-ID (corrected) 80.7%, SVM (balanced) 83.0%, SMOTE + SVM 84.0%, IFUTSVM-ID (published code) 74.3%.

## Universum construction

- Objective: `Q_U = Law(X | a^T(X-m)=0), X ~ N(mu_min, Sigma)`.
- Training-only means and equally weighted class covariance estimates; fixed numerical ridge; no tuned strategy hyperparameters.
- Support, moments, default equivalence, and modified primals: `tests/test_generative.py`.
- Gaussian class assumption and threshold-surrogate scope: `docs/THEORY.md`.
- No population theorem of superiority over random averaging.

## Protocol and ablation

- Shared candidates per split: 40; inner folds: 3; seeds: [0, 1, 2, 3, 4].
- One-factor variants: 15; completed outer cells: 900.
- Size, IF-score, and side-two controls are separate main-effect contrasts; interactions not estimated.
- Full dataset/variant/metric means, sample SDs and paired changes: `results/ablation.csv`.
- Values below: equal-weight averages of dataset mean paired differences against random averaging.

| Variant | G-mean change pp | Balanced accuracy change pp | F1 change pp | ROC-AUC change pp |
|---|---:|---:|---:|---:|
| IFUTSVM-ID (no Universum) | +0.070100 | +0.016489 | +1.944873 | +0.827395 |
| IFUTSVM-ID (published Universum) | -0.213484 | -0.189635 | +0.102082 | +0.049620 |
| IFUTSVM-ID (corrected) | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| IFUTSVM-ID (Gaussian-noise Universum) | +0.378249 | +0.227146 | +0.109966 | +0.158763 |
| IFUTSVM-ID (conditional Gaussian) | +0.680558 | +0.525683 | -0.185079 | +1.032406 |
| IFUTSVM-ID (corrected; u × 0.25) | -1.034089 | -0.631844 | -0.436217 | -0.021093 |
| IFUTSVM-ID (corrected; u × 0.5) | -0.173769 | -0.075339 | -0.129669 | +0.300313 |
| IFUTSVM-ID (corrected; u × 2) | +0.471032 | +0.371226 | +1.013034 | +0.657934 |
| IFUTSVM-ID (corrected; IF scores off) | -0.404712 | -0.139216 | +0.628279 | +0.051043 |
| IFUTSVM-ID (corrected; paper-dual sign) | +1.334771 | +0.991851 | +0.178187 | -0.094455 |
| IFUTSVM-ID (conditional Gaussian; u × 0.25) | +0.396802 | +0.811914 | +0.749067 | +1.354896 |
| IFUTSVM-ID (conditional Gaussian; u × 0.5) | -0.120004 | +0.350183 | +0.491608 | +0.864906 |
| IFUTSVM-ID (conditional Gaussian; u × 2) | +1.504508 | +1.059185 | +0.559509 | +0.684386 |
| IFUTSVM-ID (conditional Gaussian; IF scores off) | -0.137020 | +0.156708 | -0.561962 | +0.439638 |
| IFUTSVM-ID (conditional Gaussian; paper-dual sign) | +2.312505 | +1.670329 | -1.893407 | +1.069785 |

## Pre-specified theory outcome

- Original synthetic records audited: 336; transfer supported: False.
- High-IR balanced-accuracy change: +0.015277778 pp.
- Absolute-boundary-distance change: +0.0175812479949.
- Original registration SHA-256: `ff156fe4222f35ed88f8550853e3d6e812976ad5acd868cdc241e870032a251c`.
- All penalties/seeds retained. Detailed contrasts and absolute performance: `results/research/theory_empirical.json`.
- KEEL results are external-validity evidence, not a replacement pre-registration.

## Inference and label noise

- Statistical unit: dataset mean over five fixed seeds/folds.
- Friedman and Nemenyi rank/critical-difference data: `results/research/statistics.json`.
- Exact two-sided signed-rank enumeration; zero differences removed; fixed twelve-decimal rounding; Holm correction across every declared metric/pair contrast.
- Four identity contrasts (random averaging versus the main paper run) stay in the declared Holm family to preserve its original scope and saved values. Their p=1 contributes no evidence of a method difference. This bookkeeping is conservative; the smallest raw p already exceeds 0.05, so removing duplicates would not produce significance.
- IFUTSVM-ID (corrected) and its random-averaging ablation control are identical by design; their identity is audited, not presented as two independent competitors.
- Signed-rank interpretation assumes independent datasets and symmetric paired differences. Non-significance does not prove equivalence.
- Ecoli/yeast derivatives may be correlated: the required twelve-task statistics are accompanied by equal-weight source-family sensitivity in statistics.json. Neither analysis is a random sample of every classification domain.
- Statistical method source: [Demšar, JMLR 2006](https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf).
- Noise datasets: abalone9-18, ecoli4, haberman, pima, yeast3.
- Noise rates: [0.0, 0.05, 0.1, 0.2]; completed outer cells: 300.
- Labels flipped only in outer training, before inner tuning; outer test labels stay clean. Same flip permutation and candidates for all models.
- All noise means and paired changes: `results/label_noise.csv`.
- Saved artifact destinations: `figures/fig10_critical_difference`, `figures/fig11_ablation` and `figures/fig12_label_noise`. Order and parameters are fixed.

### Main-model omnibus and source-family sensitivity

| Scope | Protocol | Metric | Units | Friedman statistic | p | Nemenyi CD | Source |
|---|---|---|---:|---:|---:|---:|---|
| Twelve tasks | Hold-out | G-mean | 12 | 44.214285714 | 6.70293434661e-08 | 2.600173851 | `results/research/statistics.json#/friedman_nemenyi/holdout/gmean` |
| Twelve tasks | Hold-out | Balanced accuracy | 12 | 41.678571429 | 2.12799045839e-07 | 2.600173851 | `results/research/statistics.json#/friedman_nemenyi/holdout/balanced_accuracy` |
| Twelve tasks | Nested CV | G-mean | 12 | 31.359165425 | 2.16442408176e-05 | 2.600173851 | `results/research/statistics.json#/friedman_nemenyi/nested_cv/gmean` |
| Twelve tasks | Nested CV | Balanced accuracy | 12 | 33.827123696 | 7.26471526454e-06 | 2.600173851 | `results/research/statistics.json#/friedman_nemenyi/nested_cv/balanced_accuracy` |
| Source families | Hold-out | G-mean | 8 | 35.250000000 | 3.85453937347e-06 | 3.184549588 | `results/research/statistics.json#/family_sensitivity/friedman_nemenyi/holdout/gmean` |
| Source families | Hold-out | Balanced accuracy | 8 | 34.660714286 | 5.01330349872e-06 | 3.184549588 | `results/research/statistics.json#/family_sensitivity/friedman_nemenyi/holdout/balanced_accuracy` |
| Source families | Nested CV | G-mean | 8 | 27.428571429 | 0.000120340283771 | 3.184549588 | `results/research/statistics.json#/family_sensitivity/friedman_nemenyi/nested_cv/gmean` |
| Source families | Nested CV | Balanced accuracy | 8 | 31.339285714 | 2.18345006953e-05 | 3.184549588 | `results/research/statistics.json#/family_sensitivity/friedman_nemenyi/nested_cv/balanced_accuracy` |

### Exact paired comparisons

- Difference direction: first model minus second model; percentage points. All declared contrasts remain visible, including duplicate-control identities.
- Run labels identify the ablation control and the independently executed main-protocol control; they use the same corrected model.

| Scope | Metric | First model / run | Second model / run | Mean change pp | Median change pp | W | Raw p | Holm p | Significant | Source |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| Twelve tasks | G-mean | IFUTSVM-ID (conditional Gaussian) / ablation | IFUTSVM-ID (corrected) / ablation | +0.680558 | +0.432483 | 14 | 0.359375 | 1 | False | `results/research/statistics.json#/paired/comparisons/0` |
| Twelve tasks | G-mean | IFUTSVM-ID (conditional Gaussian) / ablation | IFUTSVM-ID (corrected) / main protocol | +0.680558 | +0.432483 | 14 | 0.359375 | 1 | False | `results/research/statistics.json#/paired/comparisons/1` |
| Twelve tasks | G-mean | IFUTSVM-ID (corrected) / ablation | IFUTSVM-ID (corrected) / main protocol | +0.000000 | +0.000000 | 0 | 1 | 1 | False | `results/research/statistics.json#/paired/comparisons/2` |
| Twelve tasks | Balanced accuracy | IFUTSVM-ID (conditional Gaussian) / ablation | IFUTSVM-ID (corrected) / ablation | +0.525683 | +0.486644 | 14 | 0.359375 | 1 | False | `results/research/statistics.json#/paired/comparisons/3` |
| Twelve tasks | Balanced accuracy | IFUTSVM-ID (conditional Gaussian) / ablation | IFUTSVM-ID (corrected) / main protocol | +0.525683 | +0.486644 | 14 | 0.359375 | 1 | False | `results/research/statistics.json#/paired/comparisons/4` |
| Twelve tasks | Balanced accuracy | IFUTSVM-ID (corrected) / ablation | IFUTSVM-ID (corrected) / main protocol | +0.000000 | +0.000000 | 0 | 1 | 1 | False | `results/research/statistics.json#/paired/comparisons/5` |
| Twelve tasks | F1 | IFUTSVM-ID (conditional Gaussian) / ablation | IFUTSVM-ID (corrected) / ablation | -0.185079 | +0.000000 | 20 | 0.8203125 | 1 | False | `results/research/statistics.json#/paired/comparisons/6` |
| Twelve tasks | F1 | IFUTSVM-ID (conditional Gaussian) / ablation | IFUTSVM-ID (corrected) / main protocol | -0.185079 | +0.000000 | 20 | 0.8203125 | 1 | False | `results/research/statistics.json#/paired/comparisons/7` |
| Twelve tasks | F1 | IFUTSVM-ID (corrected) / ablation | IFUTSVM-ID (corrected) / main protocol | +0.000000 | +0.000000 | 0 | 1 | 1 | False | `results/research/statistics.json#/paired/comparisons/8` |
| Twelve tasks | ROC-AUC | IFUTSVM-ID (conditional Gaussian) / ablation | IFUTSVM-ID (corrected) / ablation | +1.032406 | +0.339961 | 9 | 0.064453125 | 0.7734375 | False | `results/research/statistics.json#/paired/comparisons/9` |
| Twelve tasks | ROC-AUC | IFUTSVM-ID (conditional Gaussian) / ablation | IFUTSVM-ID (corrected) / main protocol | +1.032406 | +0.339961 | 9 | 0.064453125 | 0.7734375 | False | `results/research/statistics.json#/paired/comparisons/10` |
| Twelve tasks | ROC-AUC | IFUTSVM-ID (corrected) / ablation | IFUTSVM-ID (corrected) / main protocol | +0.000000 | +0.000000 | 0 | 1 | 1 | False | `results/research/statistics.json#/paired/comparisons/11` |
| Source families | G-mean | IFUTSVM-ID (conditional Gaussian) / ablation | IFUTSVM-ID (corrected) / ablation | +1.257120 | +0.855708 | 9 | 0.46875 | 1 | False | `results/research/statistics.json#/family_sensitivity/paired/comparisons/0` |
| Source families | G-mean | IFUTSVM-ID (conditional Gaussian) / ablation | IFUTSVM-ID (corrected) / main protocol | +1.257120 | +0.855708 | 9 | 0.46875 | 1 | False | `results/research/statistics.json#/family_sensitivity/paired/comparisons/1` |
| Source families | G-mean | IFUTSVM-ID (corrected) / ablation | IFUTSVM-ID (corrected) / main protocol | +0.000000 | +0.000000 | 0 | 1 | 1 | False | `results/research/statistics.json#/family_sensitivity/paired/comparisons/2` |
| Source families | Balanced accuracy | IFUTSVM-ID (conditional Gaussian) / ablation | IFUTSVM-ID (corrected) / ablation | +0.876970 | +0.701411 | 9 | 0.46875 | 1 | False | `results/research/statistics.json#/family_sensitivity/paired/comparisons/3` |
| Source families | Balanced accuracy | IFUTSVM-ID (conditional Gaussian) / ablation | IFUTSVM-ID (corrected) / main protocol | +0.876970 | +0.701411 | 9 | 0.46875 | 1 | False | `results/research/statistics.json#/family_sensitivity/paired/comparisons/4` |
| Source families | Balanced accuracy | IFUTSVM-ID (corrected) / ablation | IFUTSVM-ID (corrected) / main protocol | +0.000000 | +0.000000 | 0 | 1 | 1 | False | `results/research/statistics.json#/family_sensitivity/paired/comparisons/5` |
| Source families | F1 | IFUTSVM-ID (conditional Gaussian) / ablation | IFUTSVM-ID (corrected) / ablation | +0.973628 | +0.167540 | 12 | 0.8125 | 1 | False | `results/research/statistics.json#/family_sensitivity/paired/comparisons/6` |
| Source families | F1 | IFUTSVM-ID (conditional Gaussian) / ablation | IFUTSVM-ID (corrected) / main protocol | +0.973628 | +0.167540 | 12 | 0.8125 | 1 | False | `results/research/statistics.json#/family_sensitivity/paired/comparisons/7` |
| Source families | F1 | IFUTSVM-ID (corrected) / ablation | IFUTSVM-ID (corrected) / main protocol | +0.000000 | +0.000000 | 0 | 1 | 1 | False | `results/research/statistics.json#/family_sensitivity/paired/comparisons/8` |
| Source families | ROC-AUC | IFUTSVM-ID (conditional Gaussian) / ablation | IFUTSVM-ID (corrected) / ablation | +1.156848 | +0.213860 | 6 | 0.21875 | 1 | False | `results/research/statistics.json#/family_sensitivity/paired/comparisons/9` |
| Source families | ROC-AUC | IFUTSVM-ID (conditional Gaussian) / ablation | IFUTSVM-ID (corrected) / main protocol | +1.156848 | +0.213860 | 6 | 0.21875 | 1 | False | `results/research/statistics.json#/family_sensitivity/paired/comparisons/10` |
| Source families | ROC-AUC | IFUTSVM-ID (corrected) / ablation | IFUTSVM-ID (corrected) / main protocol | +0.000000 | +0.000000 | 0 | 1 | 1 | False | `results/research/statistics.json#/family_sensitivity/paired/comparisons/11` |

### Label-noise aggregate

- Percent means and sample SD across dataset means, after averaging the fixed seeds within each dataset. This SD describes dataset heterogeneity; it is not a confidence interval.
- Each source row is `results/research/summary.json#/noise/datasets/<dataset>/<rate>/<model>/metrics/<metric>`; all dataset/seed summaries are retained in `results/label_noise.csv`.

| Training-label flips | Model | G-mean mean ± SD | Balanced accuracy mean ± SD | F1 mean ± SD | ROC-AUC mean ± SD |
|---:|---|---:|---:|---:|---:|
| 0% | IFUTSVM-ID (corrected) | 73.546617 ± 14.887466 | 76.208683 ± 12.439514 | 59.616743 ± 18.764788 | 85.050186 ± 12.830980 |
| 0% | IFUTSVM-ID (conditional Gaussian) | 73.550739 ± 11.648908 | 76.201805 ± 10.449078 | 58.424788 ± 13.159112 | 85.715159 ± 12.641560 |
| 0% | SVM (balanced) | 79.000181 ± 13.460696 | 79.530335 ± 12.868472 | 60.033756 ± 17.175048 | 86.832900 ± 12.286373 |
| 5% | IFUTSVM-ID (corrected) | 73.062198 ± 16.880978 | 75.718815 ± 13.796242 | 56.053771 ± 18.371970 | 82.769656 ± 14.081531 |
| 5% | IFUTSVM-ID (conditional Gaussian) | 76.521849 ± 14.373799 | 77.879421 ± 12.846282 | 55.747169 ± 17.553403 | 84.663519 ± 12.214297 |
| 5% | SVM (balanced) | 78.172208 ± 14.182930 | 78.705750 ± 13.633177 | 56.912702 ± 18.460323 | 85.852614 ± 12.949609 |
| 10% | IFUTSVM-ID (corrected) | 70.439335 ± 14.363834 | 73.759263 ± 11.819339 | 54.393674 ± 19.526626 | 82.067946 ± 11.009445 |
| 10% | IFUTSVM-ID (conditional Gaussian) | 69.644513 ± 13.087609 | 73.798562 ± 11.587253 | 51.458775 ± 19.108366 | 83.305852 ± 12.143176 |
| 10% | SVM (balanced) | 78.048857 ± 13.525157 | 78.615250 ± 12.756935 | 54.194516 ± 18.570842 | 84.873447 ± 12.672989 |
| 20% | IFUTSVM-ID (corrected) | 70.730380 ± 14.455100 | 72.706200 ± 12.306323 | 48.062506 ± 19.141929 | 80.543721 ± 13.897562 |
| 20% | IFUTSVM-ID (conditional Gaussian) | 69.914062 ± 15.141409 | 72.364013 ± 12.652077 | 47.935722 ± 19.488330 | 79.206393 ± 14.443403 |
| 20% | SVM (balanced) | 69.738619 ± 10.654301 | 72.510349 ± 9.711453 | 42.153372 ± 21.315537 | 81.198219 ± 12.577409 |
