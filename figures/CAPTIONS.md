# Figure captions and sources

## fig01_methodology_block_diagram

Scaling, Universum construction and fuzzy scores use training rows only; inner validation chooses parameters.

- Source: `results/summary.json` `/config`
- Command: `python -m experiments.presentation_figures`.

## fig02_model_architecture

Reduced kernel basis D = [X1; X2*]; default symmetric Universum constraints; all kernel blocks share bandwidth.

- Source: `results/summary.json` `/config`
- Command: `python -m experiments.presentation_figures`.

## fig03_evaluation_protocol

Outer test data never enter tuning, scaling, SMOTE or Universum fitting. Main selection: balanced accuracy.

- Source: `results/summary.json` `/config`
- Command: `python -m experiments.presentation_figures`.

## fig04_main_performance

Holdout balanced accuracy: IFUTSVM-ID (corrected) 80.7%, SVM (balanced) 83.0%, SMOTE + SVM 84.0%. Equal-weight means over 12 tasks; bars show SD between dataset means.

- Source: `results/summary.json` `/headline`, `/protocols/holdout/aggregate`, `/protocols/nested_cv/aggregate`
- Command: `python -m experiments.presentation_figures`.

## fig05_roc_curves

Minimum, lower-median and maximum dataset IR; seed zero. Curves use persisted minority-positive decision scores, with no performance-based dataset choice.

- Source: `results/summary.json` `/artifact_plan/representative_datasets`, `/presentation_illustrations`
- Command: `python -m experiments.presentation_figures`.

## fig06_confusion_matrices

IFUTSVM-ID (corrected) versus SMOTE + SVM (best complete-benchmark mean balanced accuracy). Counts summed over 5 seeds; cell shade and percent are row recall (0-100%).

- Source: `results/summary.json` `/presentation_illustrations`, `/protocols/holdout/aggregate`
- Command: `python -m experiments.presentation_figures`.

## fig07_learning_curves

abalone9-18; parameters fixed at original training-selected values. Mean and SD over 3 inner folds; outer test rows excluded. Descriptive curves, not new tuning.

- Source: `results/summary.json` `/course_curves/plan/learning`, `/course_curves/groups`
- Command: `python -m experiments.presentation_figures`.

## fig08_validation_curves

IFUTSVM-ID (corrected), abalone9-18; vary one parameter, with all others fixed at original training-selected values. Mean and SD across inner folds; descriptive reuse of inner selection, not an unbiased test estimate.

- Source: `results/summary.json` `/course_curves/plan/validation`, `/course_curves/groups/paper`
- Command: `python -m experiments.presentation_figures`.

## fig09_reproduction

IFUTSVM-ID (corrected): accuracy-selected holdout means; balanced accuracy uses the same predictions. Explanation categories do not identify undocumented author settings.

- Source: `results/summary.json` `/reproduction/datasets`
- Command: `python -m experiments.presentation_figures`.

## fig10_critical_difference

Friedman/Nemenyi across all tasks; each predeclared protocol is a separate test family. The ruler shows the critical rank difference; post-hoc inference uses the omnibus gate.

- Source: `results/summary.json` `/main_statistics/friedman_nemenyi`
- Command: `python -m experiments.presentation_figures`.

## fig11_ablation

Paired differences versus IFUTSVM-ID (corrected): equal-weight mean and SD across 12 dataset means. Every pre-specified variant is retained.

- Source: `results/summary.json` `/research/ablation`, `/research/plan/variants`
- Command: `python -m experiments.presentation_figures`.

## fig12_label_noise

Only training labels are flipped; outer test labels remain clean. Identical flip permutations and tuning budgets for all models. Mean and SD across the fixed holdout seeds.

- Source: `results/summary.json` `/research/noise`, `/research/plan/noise_levels`, `/research/plan/noise_models`
- Command: `python -m experiments.presentation_figures`.
