# Experiment protocol and evidence layout

`python -m experiments.run_all` performs the main benchmark. The effective
budget, seed entropy and fit-only compute rationale are generated in
`results/compute_budget.json`. `results/experiment_config.json` records the
committed revision, source fingerprint, dependency versions and selected models.

## Splitting and selection

Protocol A uses stratified 70/30 holdouts with the issued seeds. Protocol B uses
the issued shuffled stratified outer folds. Both use three stratified inner folds on
outer training rows only. The exact index order is persisted; models receive
identical arrays. Candidate lists are shared across models for each split.
Irrelevant candidate fields are ignored, never redrawn. The stored `parameters`
object is this common proposal: SVM uses C and mu, faithful mode uses C, mu and
eps, TSVM/UTSVM use their declared subset, and paper mode uses all fields. Fixed
baseline ridge and paper-side convention are defined in `docs/BASELINES.md`.

Main selection maximizes mean inner balanced accuracy. Accuracy selects the
separate paper reproduction. Exact ties use the earliest candidate. Failed
training fits consume budget and remain in the cache. A candidate is valid only
when all its inner fits succeed. If the selected final training fit fails,
the next training-ranked candidate is tried. A failure during outer evaluation
does not trigger a different selection. No successful outer metric changes a
candidate, threshold, seed, model convention or stopping decision.

## Preprocessing and scores

The loader preserves official raw rows and nominal values. Nominal encoding
is fitted in each training pipeline; unseen categories are ignored. Scaling
follows encoding and precedes model fitting. Faithful raw omits scaling but
still requires nominal encoding. SMOTE occurs after train-only preprocessing
inside the pipeline, never before splitting. Its neighbor count is bounded by
the actual minority training count. All model randomness uses the split seed.

The positive class is the training minority (the sorted first label on ties).
Continuous sklearn decision scores are reoriented to that label before ROC-AUC.
Confusion matrices use negative, positive order. Precision, recall and F1 use
zero for zero denominators. G-mean uses recall and specificity. Accuracy is
ordinary correct-count accuracy, distinct from the faithful demo return value.

## Persistence, reuse and auditing

Every attempted inner fit records train/validation indices, parameters and
metadata. Successful fits include scaler statistics, labels, predictions,
scores, all metrics and ROC points. Errors include stage and elapsed fit time.
Typed Parquet columns preserve original row indices and each evaluation vector.
The main and reproduction selection can reference the same outer candidate;
its test predictions are reused. Published-configuration probes and subsequent
fixed diagnostics also reuse identical completed configurations where present.

Caches require matching source, data, versions, parameters and exact indices.
Each dataset/model/protocol has one Parquet table; a JSON index describes all
groups. Disposable SQLite cache under tmp checkpoints individual fits and can be
restored from Parquet. Table and index writes are atomic. Execution records retain timing segments on resume;
reported fitting time sums the measured segments. The shared candidate budget is recorded in `results/compute_budget.json`.
TSVM and UTSVM use the reduced basis described in `docs/BASELINES.md`.

`python -m experiments.audit` reconstructs the split plan and every training
preprocessor from original data, checks equal budgets and selection, and
independently rederives metrics with sklearn. Its corruption test verifies that
changed scaler evidence is rejected. `python -m experiments.refit` fits a
seeded random sample of completed runs from scratch and checks predictions and
scores. No pickles are required.

## Reporting

`python -m experiments.summarize` requires every dataset/model group to contain
all issued outer runs. It writes the canonical summary and CSV tables with
sample standard deviations, computed per-dataset ranks, average ranks across
datasets and computed best markers. Time ranks prefer lower values; performance
ranks prefer higher values. Displayed performance metrics are percentages;
underlying JSON/CSV means remain fractions. Time is in seconds. The partial
mode writes `progress_summary.json`, never a final table.

Reproduction probes are fixed in `results/reproduction_probe_plan.json` before
outcomes are interpreted. They measure implementation differences, not a
test-selected winning configuration. The generated reproduction notes retain
unmatched published numbers and explicitly distinguish measured sensitivity
from identifying the authors' unavailable configuration.
