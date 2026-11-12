# IFUTSVM-ID

<!-- MAIN FINDING -->
IFUTSVM-ID (corrected) does not beat the simple imbalance-aware baselines under the shared tuning protocol on these datasets. Equal-weight hold-out balanced accuracy over 12 datasets: IFUTSVM-ID (corrected) 80.7%, SVM (balanced) 83.0%, SMOTE + SVM 84.0%, IFUTSVM-ID (published code) 74.3%.
<!-- END MAIN FINDING -->

Python implementation of IFUTSVM-ID with executable-faithful and corrected
estimators, a conditional Gaussian Universum, and comparisons on KEEL data.
The mathematical conventions are in [SPECIFICATION](docs/SPECIFICATION.md).

## Run the notebook

From the repository root, use Python 3.12 and run the following commands in PowerShell:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m experiments.notebook --execute --output tmp/notebook.ipynb
```

The generated notebook can also be opened in Jupyter and run from top to
bottom with the repository root as its working directory. It includes the
kernel, fuzzy scores, Universum constructions, primal problems, prediction
rule and plotting code. The small-data example performs fresh inner tuning
under both evaluation protocols. Loading and execution are offline after
dependency installation.

For Python 3.10 or later, `pip install -e '.[dev,artifacts]'` installs compatible
dependency ranges. `requirements.txt` pins the primary Python 3.12 environment.

## Package API

```python
from ifutsvm import IFUTSVMID
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ifutsvm.data import load

X, y = load('glass4')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0)
model = make_pipeline(StandardScaler(), IFUTSVMID(random_state=0))
# Fit and tune using training rows only.
model.fit(X_train, y_train)
predictions = model.predict(X_test)
scores = model.decision_function(X_test)
```

The positive decision score favors `classes_[1]`. For minority-class ROC-AUC,
orient scores using the minority label defined by training counts. A callable
Universum strategy receives `(X_minority, X_majority, count, rng)`. The
`ConditionalGaussianUniversum` strategy is available from `ifutsvm.universum`.

`ifutsvm.reference.trace_reference(train, test, c, mu, eps)` takes a final +/-1
label column and preserves the MATLAB demo's executable behavior, including
unsupported sampling errors. Its returned accuracy is balanced accuracy;
the corrected API uses the conventional metrics in the specification.

## Data, results and figures

- `data/keel/`: complete official data and source URL/checksum manifest.
- `results/summary.json`: dataset metrics, statistical comparisons, curves and
  scientific interpretation. Numeric arrays feed all figures and tables.
- `results/presentation_predictions.json`: persisted labels and scores for the
  ROC and confusion comparisons; all specified seeds are included.
- `results/tables/`: CSV and LaTeX comparisons, hardware and software versions.
- `figures/`: PNG/PDF figures and editable draw.io diagrams.
- `src/ifutsvm/`, `tests/`: package implementation and numerical/API contracts.

Regenerate the figures and tables from the saved summary:

```powershell
.venv/Scripts/python -m experiments.presentation_figures
.venv/Scripts/python -m experiments.course_tables
.venv/Scripts/python -m experiments.solver_crosscheck
```

## Recompute experiments

The complete benchmark is substantially longer than notebook execution.
Use a new output directory for fresh fits; completed runs are resumed only
when source, parameters, data, splits and dependencies match.

```powershell
.venv/Scripts/python -m experiments.run_all --output tmp/recompute-results
.venv/Scripts/python -m experiments.audit --output tmp/recompute-results
.venv/Scripts/python -m experiments.summarize --output tmp/recompute-results
```

The experiment scripts also include the research ablations, noise study,
reproduction diagnostics and theory check. See [EXPERIMENT_PROTOCOL](docs/EXPERIMENT_PROTOCOL.md),
[REPRODUCTION](docs/REPRODUCTION.md) and [RESEARCH](docs/RESEARCH.md) for the fixed
designs and commands. Full per-fit records are created by these runs; the
compact archive contains the numerical summaries and illustrative predictions.

Run the research experiments and theory check in new directories:

```powershell
.venv/Scripts/python -m experiments.research_run --output tmp/recompute-results/research --wave ablation
.venv/Scripts/python -m experiments.research_run --output tmp/recompute-results/research --wave noise
.venv/Scripts/python -m experiments.theory_check --output tmp/recompute-results/research
.venv/Scripts/python -m experiments.theory_report --output tmp/recompute-results/research --audit-records
.venv/Scripts/python -m experiments.research_report --output tmp/recompute-results
```

## Scientific scope

The conditional Gaussian construction has a density-conditioning objective and
a proved population identity under its stated assumptions. Its threshold
surrogate is not a theorem about IFUTSVM predictive performance. The original
synthetic transfer prediction was refuted; all penalties and seeds remain in
the recorded analysis. Across-dataset non-significance does not establish
equivalence. Related KEEL tasks receive a source-family sensitivity analysis.

[ALGORITHM](docs/ALGORITHM.md) gives the equations and steps,
[THEORY](docs/THEORY.md) gives the assumptions and proof,
[BASELINES](docs/BASELINES.md) defines the comparison models, and
[PORT_DISCREPANCIES](docs/PORT_DISCREPANCIES.md) records paper/code differences.

## References and license

M. Tanveer and A. Quadir, *Robust Universum Twin Support Vector Machine for
Imbalanced Data*, [arXiv:2410.20335](https://arxiv.org/abs/2410.20335).
The [MATLAB reference](https://github.com/abdulquadir1999/IFUTSVM-ID) is not
redistributed. Bibliographic entries are in `references.bib`; package citation
metadata is in `CITATION.cff`. See `LICENSE` and [LICENSING](docs/LICENSING.md).
