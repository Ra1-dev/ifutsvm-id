# Reproduction evidence

Recompute these comparisons with `python -m experiments.reproduction_report` from persisted predictions, the fixed diagnostic plan, and Table S.II.

Table S.II supplies C, Cu, epsilon and bandwidth but omits the independent C3. Its split indices, random seed and complete preprocessing are unavailable. We use the stated default for the missing parameter and do not infer it from test scores.

Main reproduction uses accuracy-selected inner CV on each prescribed stratified holdout. Reference-configuration diagnostics use the published parameters and fixed seeds. The separately pre-specified yeast1 accuracy-ceiling diagnostic tunes four other model families on accuracy. None changes the main comparison, which uses balanced accuracy for selection. Metrics are ordinary accuracy unless explicitly named otherwise.

Accuracy means within the prescribed tolerance: 6/12. Values outside tolerance are not claimed as reproduced.

IFUTSVM-ID (corrected) does not beat the simple imbalance-aware baselines under the shared tuning protocol on these datasets. Equal-weight hold-out balanced accuracy over 12 datasets: IFUTSVM-ID (corrected) 80.7%, SVM (balanced) 83.0%, SMOTE + SVM 84.0%, IFUTSVM-ID (published code) 74.3%.

| Dataset | Published ACC % | Reproduction mean ± std % | Gap pp | Published config, seed 0 % | Within tolerance |
|---|---:|---:|---:|---:|---|
| abalone9-18 | 89.9500 | 92.5455 ± 2.7761 | +2.5955 | 90.9091 | False |
| ecoli-0-1_vs_5 | 97.2200 | 97.5000 ± 1.8109 | +0.2800 | 91.6667 | True |
| ecoli3 | 91.0000 | 90.6931 ± 1.9301 | -0.3069 | 92.0792 | True |
| ecoli4 | 96.0000 | 97.6238 ± 1.6568 | +1.6238 | 98.0198 | True |
| glass4 | 91.9800 | 96.6154 ± 2.0059 | +4.6354 | 95.3846 | False |
| haberman | 75.8200 | 76.3043 ± 1.4172 | +0.4843 | 78.2609 | True |
| new-thyroid1 | 96.8800 | 97.2308 ± 2.0059 | +0.3508 | 100.0000 | True |
| pima | 71.7400 | 73.6797 ± 3.2597 | +1.9397 | 73.1602 | True |
| vehicle1 | 69.9600 | 80.9449 ± 1.4888 | +10.9849 | 74.4094 | False |
| yeast1 | 92.1400 | 76.4574 ± 1.4531 | -15.6826 | 73.9910 | False |
| yeast-2_vs_4 | 79.2200 | 94.7097 ± 1.0601 | +15.4897 | 92.9032 | False |
| yeast3 | 89.4400 | 95.0673 ± 0.5258 | +5.6273 | 94.6188 | False |

## Controlled diagnostic effects

Changes below are relative to the published-parameter, standardized, symmetric, reduced-basis probe on matching seeds. A protocol contrast may change test rows and is labeled accordingly. Full-basis and normalization diagnostics have a single predeclared seed. Missing or failed probes are shown, never replaced by favorable cases. Effects can identify sensitivity to implementation choices; they do not establish which choice produced the authors number.

### abalone9-18

Official raw features: 8; Table S.II features: 7. Full-data majority-class rate: 94.2544%. Unreported C3 default: 1.

| Fixed variant | Valid/expected | ACC mean % | Balanced ACC mean % | ΔACC pp | Δbalanced accuracy pp | Same test rows | Register |
|---|---:|---:|---:|---:|---:|---|---|
| Published parameters, scaled | 5/5 | 91.5455 | 55.8566 | 0.0000 | 0.0000 | True | DSC-34 |
| Raw features | 5/5 | 91.8182 | 53.8387 | 0.2727 | -2.0178 | True | DSC-01, DSC-02 |
| Printed dual sign | 5/5 | 64.7273 | 67.5585 | -26.8182 | 11.7020 | True | DSC-11 |
| Printed primal sign | 5/5 | 91.5455 | 55.8566 | 0.0000 | 0.0000 | True | DSC-11 |
| Executable ridge | 5/5 | 92.6364 | 51.3898 | 1.0909 | -4.4667 | True | DSC-09, DSC-34 |
| Unstratified split | 5/5 | 91.7273 | 56.0149 | 0.1818 | 0.1583 | False | DSC-29 |
| Full kernel basis | 1/1 | 87.2727 | 64.3999 | -3.6364 | 1.6722 | True | DSC-30 |
| RKHS normalization | 1/1 | 90.0000 | 65.8491 | -0.9091 | 3.1215 | True | DSC-31 |
| Nominal feature omitted | 1/1 | 94.0909 | 57.2092 | 3.1818 | -5.5184 | True | DSC-38 |

Status: not reproduced. The table above measures effects of the pre-specified ambiguities and differences. The authors exact configuration remains unidentified; any residual difference is not assigned a fabricated cause.

Measured configuration decomposition: changing from the fixed published-parameter probe to the training-selected configurations shifts the mean by +1.0000 pp. The published-parameter mean is 91.5455%, leaving +1.5955 pp against the paper, within tolerance. This contrast measures the effect of configuration choice, including the explicit missing-C3 convention (DSC-34); it does not identify the authors unpublished settings.

Metric-definition control (DSC-13): the same accuracy-selected configurations average 92.5455% ordinary accuracy and 54.9461% balanced accuracy. The executable returns the latter under its accuracy name. This establishes the size of the metric discrepancy on these outputs, not which metric the authors used for this table row.

### ecoli-0-1_vs_5

Official raw features: 6; Table S.II features: 6. Full-data majority-class rate: 91.6667%. Unreported C3 default: 1.

| Fixed variant | Valid/expected | ACC mean % | Balanced ACC mean % | ΔACC pp | Δbalanced accuracy pp | Same test rows | Register |
|---|---:|---:|---:|---:|---:|---|---|
| Published parameters, scaled | 5/5 | 91.6667 | 50.0000 | 0.0000 | 0.0000 | True | DSC-34 |
| Raw features | 5/5 | 95.8333 | 76.5152 | 4.1667 | 26.5152 | True | DSC-01, DSC-02 |
| Printed dual sign | 5/5 | 91.6667 | 50.0000 | 0.0000 | 0.0000 | True | DSC-11 |
| Printed primal sign | 5/5 | 91.6667 | 50.0000 | 0.0000 | 0.0000 | True | DSC-11 |
| Executable ridge | 5/5 | 94.4444 | 69.6970 | 2.7778 | 19.6970 | True | DSC-09, DSC-34 |
| Unstratified split | 5/5 | 91.3889 | 50.0000 | -0.2778 | 0.0000 | False | DSC-29 |
| Full kernel basis | 1/1 | 91.6667 | 50.0000 | 0.0000 | 0.0000 | True | DSC-30 |
| RKHS normalization | 1/1 | 91.6667 | 50.0000 | 0.0000 | 0.0000 | True | DSC-31 |

Metric-definition control (DSC-13): the same accuracy-selected configurations average 97.5000% ordinary accuracy and 86.5152% balanced accuracy. The executable returns the latter under its accuracy name. This establishes the size of the metric discrepancy on these outputs, not which metric the authors used for this table row.

### ecoli3

Official raw features: 7; Table S.II features: 7. Full-data majority-class rate: 89.5833%. Unreported C3 default: 1.

| Fixed variant | Valid/expected | ACC mean % | Balanced ACC mean % | ΔACC pp | Δbalanced accuracy pp | Same test rows | Register |
|---|---:|---:|---:|---:|---:|---|---|
| Published parameters, scaled | 5/5 | 90.8911 | 84.5152 | 0.0000 | 0.0000 | True | DSC-34 |
| Raw features | 5/5 | 86.9307 | 82.2929 | -3.9604 | -2.2222 | True | DSC-01, DSC-02 |
| Printed dual sign | 5/5 | 90.8911 | 84.5152 | 0.0000 | 0.0000 | True | DSC-11 |
| Printed primal sign | 5/5 | 90.8911 | 84.5152 | 0.0000 | 0.0000 | True | DSC-11 |
| Executable ridge | 5/5 | 91.8812 | 76.2929 | 0.9901 | -8.2222 | True | DSC-09, DSC-34 |
| Unstratified split | 5/5 | 92.8713 | 86.5132 | 1.9802 | 1.9980 | False | DSC-29 |
| Full kernel basis | 1/1 | 92.0792 | 87.5758 | 0.0000 | 0.0000 | True | DSC-30 |
| RKHS normalization | 1/1 | 92.0792 | 91.5657 | 0.0000 | 3.9899 | True | DSC-31 |

Metric-definition control (DSC-13): the same accuracy-selected configurations average 90.6931% ordinary accuracy and 78.0202% balanced accuracy. The executable returns the latter under its accuracy name. This establishes the size of the metric discrepancy on these outputs, not which metric the authors used for this table row.

### ecoli4

Official raw features: 7; Table S.II features: 7. Full-data majority-class rate: 94.0476%. Unreported C3 default: 1.

| Fixed variant | Valid/expected | ACC mean % | Balanced ACC mean % | ΔACC pp | Δbalanced accuracy pp | Same test rows | Register |
|---|---:|---:|---:|---:|---:|---|---|
| Published parameters, scaled | 5/5 | 98.4158 | 89.7895 | 0.0000 | 0.0000 | True | DSC-34 |
| Raw features | 5/5 | 98.0198 | 89.5789 | -0.3960 | -0.2105 | True | DSC-01, DSC-02 |
| Printed dual sign | 5/5 | 98.0198 | 89.5789 | -0.3960 | -0.2105 | True | DSC-11 |
| Printed primal sign | 5/5 | 98.4158 | 89.7895 | 0.0000 | 0.0000 | True | DSC-11 |
| Executable ridge | 5/5 | 98.2178 | 88.1228 | -0.1980 | -1.6667 | True | DSC-09, DSC-34 |
| Unstratified split | 5/5 | 98.2178 | 90.2087 | -0.1980 | 0.4193 | False | DSC-29 |
| Full kernel basis | 1/1 | 98.0198 | 91.1404 | 0.0000 | 0.0000 | True | DSC-30 |
| RKHS normalization | 1/1 | 98.0198 | 91.1404 | 0.0000 | 0.0000 | True | DSC-31 |

Metric-definition control (DSC-13): the same accuracy-selected configurations average 97.6238% ordinary accuracy and 86.2456% balanced accuracy. The executable returns the latter under its accuracy name. This establishes the size of the metric discrepancy on these outputs, not which metric the authors used for this table row.

### glass4

Official raw features: 9; Table S.II features: 9. Full-data majority-class rate: 93.9252%. Unreported C3 default: 1.

| Fixed variant | Valid/expected | ACC mean % | Balanced ACC mean % | ΔACC pp | Δbalanced accuracy pp | Same test rows | Register |
|---|---:|---:|---:|---:|---:|---|---|
| Published parameters, scaled | 5/5 | 93.8462 | 94.3852 | 0.0000 | 0.0000 | True | DSC-34 |
| Raw features | 5/5 | 94.1538 | 89.8770 | 0.3077 | -4.5082 | True | DSC-01, DSC-02 |
| Printed dual sign | 5/5 | 94.7692 | 92.5410 | 0.9231 | -1.8443 | True | DSC-11 |
| Printed primal sign | 5/5 | 93.8462 | 94.3852 | 0.0000 | 0.0000 | True | DSC-11 |
| Executable ridge | 5/5 | 95.6923 | 67.3361 | 1.8462 | -27.0492 | True | DSC-09, DSC-34 |
| Unstratified split | 5/5 | 92.3077 | 93.6171 | -1.5385 | -0.7682 | False | DSC-29 |
| Full kernel basis | 1/1 | 96.9231 | 86.6803 | 1.5385 | 0.8197 | True | DSC-30 |
| RKHS normalization | 1/1 | 95.3846 | 85.8607 | 0.0000 | 0.0000 | True | DSC-31 |

Table arithmetic: nearest possible ordinary accuracy for either rounded holdout size differs by 0.2075 pp. This checks the single-split interpretation only; it does not explain a larger performance gap or rule out an unstated aggregation.

Status: not reproduced. The table above measures effects of the pre-specified ambiguities and differences. The authors exact configuration remains unidentified; any residual difference is not assigned a fabricated cause.

Measured configuration decomposition: changing from the fixed published-parameter probe to the training-selected configurations shifts the mean by +2.7692 pp. The published-parameter mean is 93.8462%, leaving +1.8662 pp against the paper, within tolerance. This contrast measures the effect of configuration choice, including the explicit missing-C3 convention (DSC-34); it does not identify the authors unpublished settings.

Metric-definition control (DSC-13): the same accuracy-selected configurations average 96.6154% ordinary accuracy and 79.5082% balanced accuracy. The executable returns the latter under its accuracy name. This establishes the size of the metric discrepancy on these outputs, not which metric the authors used for this table row.

### haberman

Official raw features: 3; Table S.II features: 3. Full-data majority-class rate: 73.5294%. Unreported C3 default: 1.

| Fixed variant | Valid/expected | ACC mean % | Balanced ACC mean % | ΔACC pp | Δbalanced accuracy pp | Same test rows | Register |
|---|---:|---:|---:|---:|---:|---|---|
| Published parameters, scaled | 5/5 | 75.6522 | 62.2304 | 0.0000 | 0.0000 | True | DSC-34 |
| Raw features | 5/5 | 75.6522 | 64.1176 | -0.0000 | 1.8873 | True | DSC-01, DSC-02 |
| Printed dual sign | 5/5 | 75.6522 | 67.0833 | 0.0000 | 4.8529 | True | DSC-11 |
| Printed primal sign | 5/5 | 75.6522 | 62.2304 | 0.0000 | 0.0000 | True | DSC-11 |
| Executable ridge | 5/5 | 72.6087 | 67.4510 | -3.0435 | 5.2206 | True | DSC-09, DSC-34 |
| Unstratified split | 5/5 | 75.4348 | 62.6143 | -0.2174 | 0.3839 | False | DSC-29 |
| Full kernel basis | 1/1 | 79.3478 | 68.5049 | 1.0870 | 2.0833 | True | DSC-30 |
| RKHS normalization | 1/1 | 78.2609 | 66.4216 | 0.0000 | 0.0000 | True | DSC-31 |

Metric-definition control (DSC-13): the same accuracy-selected configurations average 76.3043% ordinary accuracy and 62.1324% balanced accuracy. The executable returns the latter under its accuracy name. This establishes the size of the metric discrepancy on these outputs, not which metric the authors used for this table row.

### new-thyroid1

Official raw features: 5; Table S.II features: 5. Full-data majority-class rate: 83.7209%. Unreported C3 default: 1.

| Fixed variant | Valid/expected | ACC mean % | Balanced ACC mean % | ΔACC pp | Δbalanced accuracy pp | Same test rows | Register |
|---|---:|---:|---:|---:|---:|---|---|
| Published parameters, scaled | 5/5 | 97.2308 | 91.8182 | 0.0000 | 0.0000 | True | DSC-34 |
| Raw features | 5/5 | 92.6154 | 82.5253 | -4.6154 | -9.2929 | True | DSC-01, DSC-02 |
| Printed dual sign | 5/5 | 97.2308 | 96.1616 | 0.0000 | 4.3434 | True | DSC-11 |
| Printed primal sign | 5/5 | 97.2308 | 91.8182 | 0.0000 | 0.0000 | True | DSC-11 |
| Executable ridge | 5/5 | 97.8462 | 95.0842 | 0.6154 | 3.2660 | True | DSC-09, DSC-34 |
| Unstratified split | 5/5 | 96.9231 | 93.0540 | -0.3077 | 1.2358 | False | DSC-29 |
| Full kernel basis | 1/1 | 100.0000 | 100.0000 | 0.0000 | 0.0000 | True | DSC-30 |
| RKHS normalization | 1/1 | 96.9231 | 90.9091 | -3.0769 | -9.0909 | True | DSC-31 |

Metric-definition control (DSC-13): the same accuracy-selected configurations average 97.2308% ordinary accuracy and 93.2660% balanced accuracy. The executable returns the latter under its accuracy name. This establishes the size of the metric discrepancy on these outputs, not which metric the authors used for this table row.

### pima

Official raw features: 8; Table S.II features: 8. Full-data majority-class rate: 65.1042%. Unreported C3 default: 1.

| Fixed variant | Valid/expected | ACC mean % | Balanced ACC mean % | ΔACC pp | Δbalanced accuracy pp | Same test rows | Register |
|---|---:|---:|---:|---:|---:|---|---|
| Published parameters, scaled | 5/5 | 71.4286 | 62.2691 | 0.0000 | 0.0000 | True | DSC-34 |
| Raw features | 5/5 | 69.9567 | 64.8840 | -1.4719 | 2.6148 | True | DSC-01, DSC-02 |
| Printed dual sign | 5/5 | 49.3506 | 60.3185 | -22.0779 | -1.9506 | True | DSC-11 |
| Printed primal sign | 5/5 | 71.4286 | 62.2691 | 0.0000 | 0.0000 | True | DSC-11 |
| Executable ridge | 5/5 | 74.7186 | 71.0494 | 3.2900 | 8.7802 | True | DSC-09, DSC-34 |
| Unstratified split | 5/5 | 71.4286 | 62.6104 | -0.0000 | 0.3412 | False | DSC-29 |
| Full kernel basis | 1/1 | 73.5931 | 66.3210 | 0.4329 | 2.0370 | True | DSC-30 |
| RKHS normalization | 1/1 | 73.1602 | 64.5679 | 0.0000 | 0.2840 | True | DSC-31 |

Metric-definition control (DSC-13): the same accuracy-selected configurations average 73.6797% ordinary accuracy and 70.1926% balanced accuracy. The executable returns the latter under its accuracy name. This establishes the size of the metric discrepancy on these outputs, not which metric the authors used for this table row.

### vehicle1

Official raw features: 18; Table S.II features: 18. Full-data majority-class rate: 74.3499%. Unreported C3 default: 1.

| Fixed variant | Valid/expected | ACC mean % | Balanced ACC mean % | ΔACC pp | Δbalanced accuracy pp | Same test rows | Register |
|---|---:|---:|---:|---:|---:|---|---|
| Published parameters, scaled | 5/5 | 74.4094 | 50.0000 | 0.0000 | 0.0000 | True | DSC-34 |
| Raw features | 5/5 | 75.0394 | 51.2308 | 0.6299 | 1.2308 | True | DSC-01, DSC-02 |
| Printed dual sign | 5/5 | 74.4094 | 50.0000 | 0.0000 | 0.0000 | True | DSC-11 |
| Printed primal sign | 5/5 | 74.4094 | 50.0000 | 0.0000 | 0.0000 | True | DSC-11 |
| Executable ridge | 5/5 | 83.7795 | 80.5210 | 9.3701 | 30.5210 | True | DSC-09, DSC-34 |
| Unstratified split | 5/5 | 74.4094 | 50.0000 | -0.0000 | 0.0000 | False | DSC-29 |
| Full kernel basis | 1/1 | 74.4094 | 50.0000 | 0.0000 | 0.0000 | True | DSC-30 |
| RKHS normalization | 1/1 | 74.4094 | 50.0000 | 0.0000 | 0.0000 | True | DSC-31 |

Status: not reproduced. The table above measures effects of the pre-specified ambiguities and differences. The authors exact configuration remains unidentified; any residual difference is not assigned a fabricated cause.

Measured published-configuration facts (DSC-13/DSC-34): C=100000, Cu=0.0001, epsilon=0.6, sigma=8. All 5 fixed scaled probes predict only the majority class: balanced accuracy 50.0000%, ordinary accuracy 74.4094%, versus the full-data majority rate 74.3499%. The published 69.96% is below that majority rate and equals the IF-USVM column (69.96%). These are measured facts; neither a duplication cause nor the authors split identity is established.

Metric-definition control (DSC-13): the same accuracy-selected configurations average 80.9449% ordinary accuracy and 75.4872% balanced accuracy. The executable returns the latter under its accuracy name. This establishes the size of the metric discrepancy on these outputs, not which metric the authors used for this table row.

### yeast1

Official raw features: 8; Table S.II features: 8. Full-data majority-class rate: 71.0916%. Unreported C3 default: 1.

| Fixed variant | Valid/expected | ACC mean % | Balanced ACC mean % | ΔACC pp | Δbalanced accuracy pp | Same test rows | Register |
|---|---:|---:|---:|---:|---:|---|---|
| Published parameters, scaled | 5/5 | 73.7668 | 68.5352 | 0.0000 | 0.0000 | True | DSC-34 |
| Raw features | 5/5 | 76.7265 | 68.9162 | 2.9596 | 0.3810 | True | DSC-01, DSC-02 |
| Printed dual sign | 5/5 | 70.8520 | 69.1512 | -2.9148 | 0.6160 | True | DSC-11 |
| Printed primal sign | 5/5 | 73.7668 | 68.5352 | 0.0000 | 0.0000 | True | DSC-11 |
| Executable ridge | 5/5 | 74.9776 | 64.1000 | 1.2108 | -4.4352 | True | DSC-09, DSC-34 |
| Unstratified split | 5/5 | 70.9417 | 66.4465 | -2.8251 | -2.0887 | False | DSC-29 |
| Full kernel basis | 1/1 | 73.7668 | 69.3627 | -0.2242 | 0.9916 | True | DSC-30 |
| RKHS normalization | 1/1 | 74.4395 | 70.2956 | 0.4484 | 1.9245 | True | DSC-31 |

Table arithmetic: nearest possible ordinary accuracy for either rounded holdout size differs by 0.005168539326 pp. This checks the single-split interpretation only; it does not explain a larger performance gap or rule out an unstated aggregation.

Status: not reproduced. The table above measures effects of the pre-specified ambiguities and differences. The authors exact configuration remains unidentified; any residual difference is not assigned a fabricated cause.

Published Table S.II yeast1 row (as published):

| Lower group | ACC % | Upper group | ACC % |
|---|---:|---|---:|
| IFW-LSTSVM | 71.91 | UTSVM | 91.16 |
| RUTSVM-CIL | 74.03 | IF-USVM | 91.56 |
| CGFTSVM-ID | 74.16 | IFUTSVM-ID | 92.14 |

Visible bimodality: 3 methods at 71.91–74.16%, and 3 at 91.16–92.14%; separating gap 17.00 pp. No explanation for that clustering is inferred from the row alone.

Accuracy-ceiling diagnostic fixed in advance at `23a425a7a0c9c32990a5e2a980f54fb7294d5f6c`; plan SHA-256 `f3e121b5a9614bf502a84b3252297738a5fd816b455607e9b4eae2a4e4480ee0`. Official KEEL file, protocol A, 5 seeds, 40 candidates per family/split, 3 inner folds; StandardScaler fitted on training rows only; accuracy selection with first-candidate tie breaking. Threshold fixed before fits: any family mean >= 90% makes our implementation suspect and leaves the reproduction question unresolved.

| Family | Mean ACC % | Sample SD pp | Best single split % |
|---|---:|---:|---:|
| SVM (RBF, accuracy diagnostic) | 77.623318 | 1.020118 | 78.475336 |
| Random forest | 78.161435 | 2.423185 | 79.820628 |
| Gradient boosting | 78.520179 | 1.473693 | 80.493274 |
| k-nearest neighbours | 77.130045 | 0.853786 | 78.475336 |

Published value unattained by this pre-specified diagnostic on the official KEEL file under protocol A; this finite probe is not a universal mathematical upper bound.

All families, splits and losses retained. This diagnostic does not alter the balanced-accuracy-selected main benchmark. Commands: `python -m experiments.accuracy_ceiling`; `python -m experiments.audit_accuracy_ceiling`. Predictions, scores, scalers, candidates and indices: `results/accuracy_ceiling/runs/*.parquet`.

Metric-definition control (DSC-13): the same accuracy-selected configurations average 76.4574% ordinary accuracy and 68.0833% balanced accuracy. The executable returns the latter under its accuracy name. This establishes the size of the metric discrepancy on these outputs, not which metric the authors used for this table row.

### yeast-2_vs_4

Official raw features: 8; Table S.II features: 8. Full-data majority-class rate: 90.0778%. Unreported C3 default: 1.

| Fixed variant | Valid/expected | ACC mean % | Balanced ACC mean % | ΔACC pp | Δbalanced accuracy pp | Same test rows | Register |
|---|---:|---:|---:|---:|---:|---|---|
| Published parameters, scaled | 5/5 | 93.0323 | 68.7619 | 0.0000 | 0.0000 | True | DSC-34 |
| Raw features | 5/5 | 94.3226 | 74.2381 | 1.2903 | 5.4762 | True | DSC-01, DSC-02 |
| Printed dual sign | 5/5 | 92.6452 | 68.5476 | -0.3871 | -0.2143 | True | DSC-11 |
| Printed primal sign | 5/5 | 92.9032 | 68.6905 | -0.1290 | -0.0714 | True | DSC-11 |
| Executable ridge | 5/5 | 92.5161 | 64.3095 | -0.5161 | -4.4524 | True | DSC-09, DSC-34 |
| Unstratified split | 5/5 | 94.8387 | 75.1261 | 1.8065 | 6.3642 | False | DSC-29 |
| Full kernel basis | 1/1 | 94.8387 | 76.3095 | 1.9355 | 12.9762 | True | DSC-30 |
| RKHS normalization | 1/1 | 92.9032 | 66.3095 | 0.0000 | 2.9762 | True | DSC-31 |

Status: not reproduced. The table above measures effects of the pre-specified ambiguities and differences. The authors exact configuration remains unidentified; any residual difference is not assigned a fabricated cause.

Leading supported but unproven hypothesis (DSC-13): the published ACC used balanced accuracy, as IFUTSVM.m returns balanced accuracy under the name accuracy. On the identical accuracy-selected predictions, ordinary accuracy exceeds that definition by +14.9002 pp. Balanced accuracy leaves +0.5895 pp against the published number, within tolerance. This supports the metric-interpretation hypothesis without proving which metric the authors used; ordinary accuracy remains unmatched.

Metric-definition control (DSC-13): the same accuracy-selected configurations average 94.7097% ordinary accuracy and 79.8095% balanced accuracy. The executable returns the latter under its accuracy name. This establishes the size of the metric discrepancy on these outputs, not which metric the authors used for this table row.

### yeast3

Official raw features: 8; Table S.II features: 8. Full-data majority-class rate: 89.0162%. Unreported C3 default: 1.

| Fixed variant | Valid/expected | ACC mean % | Balanced ACC mean % | ΔACC pp | Δbalanced accuracy pp | Same test rows | Register |
|---|---:|---:|---:|---:|---:|---|---|
| Published parameters, scaled | 5/5 | 94.6188 | 85.5282 | 0.0000 | 0.0000 | True | DSC-34 |
| Raw features | 5/5 | 92.4664 | 91.2959 | -2.1525 | 5.7677 | True | DSC-01, DSC-02 |
| Printed dual sign | 5/5 | 89.8206 | 90.5254 | -4.7982 | 4.9972 | True | DSC-11 |
| Printed primal sign | 5/5 | 94.3946 | 86.8334 | -0.2242 | 1.3052 | True | DSC-11 |
| Executable ridge | 5/5 | 93.5874 | 81.0132 | -1.0314 | -4.5150 | True | DSC-09, DSC-34 |
| Unstratified split | 5/5 | 94.2601 | 82.9516 | -0.3587 | -2.5766 | False | DSC-29 |
| Full kernel basis | 1/1 | 94.6188 | 87.1382 | 0.0000 | 2.6834 | True | DSC-30 |
| RKHS normalization | 1/1 | 94.3946 | 85.2234 | -0.2242 | 0.7685 | True | DSC-31 |

Status: not reproduced. The table above measures effects of the pre-specified ambiguities and differences. The authors exact configuration remains unidentified; any residual difference is not assigned a fabricated cause.

Leading supported but unproven hypothesis (DSC-13): the published ACC used balanced accuracy, as IFUTSVM.m returns balanced accuracy under the name accuracy. On the identical accuracy-selected predictions, ordinary accuracy exceeds that definition by +6.7827 pp. Balanced accuracy leaves -1.1554 pp against the published number, within tolerance. This supports the metric-interpretation hypothesis without proving which metric the authors used; ordinary accuracy remains unmatched.

Metric-definition control (DSC-13): the same accuracy-selected configurations average 95.0673% ordinary accuracy and 88.2846% balanced accuracy. The executable returns the latter under its accuracy name. This establishes the size of the metric discrepancy on these outputs, not which metric the authors used for this table row.

## Reproduction limits

The timing-limited search budget is recorded in results/compute_budget.json and is smaller than the paper grid. C3 is not published. The manuscript and executable disagree on kernels, signs, basis, normalization interpretation and reported metrics; the register distinguishes each. Abalone includes a nominal feature absent from the published feature count. A matched accuracy alone does not establish equivalent fitted models. Statistical variation is shown over the fixed runs, not selected seeds.

Commands: `python -m experiments.reproduction_probes`; `python -m experiments.table_audit`; `python -m experiments.reproduction_report`. Raw per-configuration predictions, scores and indices are in results/reproduction_probes/*.parquet. Main accuracy-selected outputs are reproduction records in the *__paper__holdout.parquet members of results/runs/<dataset>.zip. The evidence reader opens these bundles directly; `python -m experiments.evidence unpack` restores exact loose tables when needed.
