# Implementation specification

These conventions define the executable-faithful API, corrected model, sampling,
fuzzy scores, optimization problems and experimental protocol. They resolve
ambiguities between the paper and its reference implementation. Independent
reference-evaluation observations are identified explicitly; they are not
represented as a locally repeated Octave execution.

## 1. Roles and notation

- Minority class = the label with fewer training rows (ties: the first label in `np.unique` order). It is the **positive class** for every metric (precision, recall, F1, ROC-AUC, G-mean).
- X1 ∈ R^{m1×n}: minority rows. X2 ∈ R^{m2×n}: majority rows. m1 ≤ m2. Plane 1 is the minority plane, plane 2 the majority plane.
- `e` = vectors of ones; z = [w; b].

## 2. Reference semantics (faithful mode)

The reference is **the executable behaviour of the original files** `main.m` (lines from `[lengthA,n] = size(A);` to `U=(Au+Bu)/2;`), `IFUTSVM.m`, `non_linear_score_values.m`, `kernelfun.m`, `Evaluate.m`, executed verbatim by the independent evaluation in GNU Octave 8.4.
- **Fractional ranges used as indices** (e.g. `B(1:(m2-1)/m1:(m2-(m2-1)/m1),:)`): Octave builds the range `base:inc:limit` and rounds each element to the nearest integer (halves away from zero). MATLAB R2023a raises an error on such indices. Faithful mode follows Octave. Range lengths and index values follow Octave's floating-point tolerance exactly.
- **QP solves.** Octave's `quadprog` rejects the second QP (its Hessian `HOINVHOT` is singular PSD with eigenvalues ≈ −1e-11) and returns zeros, which MATLAB's `quadprog` does not. The independent reference evaluation therefore replaced `quadprog` by an oracle with MATLAB semantics: `min ½xᵀ((H+Hᵀ)/2)x + fᵀx, lb ≤ x ≤ ub`, solved to 1e-12 gaps. The dual solution may be non-unique, but `w1`, `w2` are unique; The faithful evaluation compares `w`, `y` and predictions, never the dual vectors. A solver must reach the faithful evaluation tolerances (default solver tolerances do not: the `ep = 1e-5` ridge makes `w` extremely sensitive).
- Fixtures flagged `meta_well_conditioned = False` are cases where two independent high-accuracy solvers disagree on `w` by >1e-6; there only predictions and deterministic arrays are compared.
- Where the reference raises (e.g. |m1 − m2| ≤ 1 gives an empty or zero index), `trace_reference` must raise.

## 3. Faithful API contract 

```python
from ifutsvm.reference import trace_reference
tr = trace_reference(train, test, c, mu, eps)   # train/test: (n, p+1) float arrays, labels ±1 in the last column
```
`tr` is a dict of numpy arrays with the **values the MATLAB variables hold when the reference finishes**:
- from the `main.m` block: `main_U`;
- from `IFUTSVM.m` (after its internal class swap and all re-assignments): `s1, s2, U, A, B, B1, C, K1, K2, K3, H, G, O, G1, GOINVGOT, HOINVHOT, f1, f2, lowb1, upb1, lowb2, upb2, w1, w2, K, y1, y2, classifier`, plus `acc_returned` (its first return value);
- from `Evaluate(A_test(:,end), yprediction, 1)`: `eval_AUC, eval_accuracy, eval_sensitivity, eval_specificity, eval_precision, eval_recall, eval_f_measure, eval_gmean`.
Column vectors may be returned 1-D. NaN must be reproduced where the reference yields NaN.

## 4. Discrepancy register format 

`docs/PORT_DISCREPANCIES.md`, one row per item: `ID | where (file:line or paper eq.) | paper says | code does | executable behaviour (Octave/MATLAB) | faithful mode | paper mode | measured impact (command + number)`. IDs `DSC-01…` are stable once assigned.

## 5. Paper mode specification 

### 5.1 Kernel
K(p, q) = exp(−‖p − q‖² / (2µ²)) everywhere: training, test and scores.

### 5.2 Sampling (`universum="random_average"`, default)
With `rng = numpy.random.default_rng(random_state)`:
- X2* : m1 rows of X2 drawn **without** replacement.
- U : u = max(m2 − m1, ⌈m1/2⌉) rows, each (x_i + x_j)/2 with x_i a uniformly drawn X1 row and x_j a uniformly drawn X2 row (pairs drawn with replacement). The paper's u = m2 − m1 is kept whenever it is ≥ ⌈m1/2⌉; the floor exists so that U* is defined for mildly imbalanced data.
- U* : g = ⌈m1/2⌉ rows of U drawn without replacement.
- D (kernel basis): `kernel_basis="reduced"` (default) → D = [X1; X2*]; `"full"` → D = [X1; X2].

### 5.3 Intuitionistic-fuzzy scores (supplementary S.I, eqs. 1–6)
Computed on the full training set (X1 ∪ X2) with the kernel of 5.1, η = `eta` (default 1e-4):
- class centre distance d_i = sqrt(max(0, 1 − 2·mean_j K(x_i, x_j) + mean_{j,k} K(x_j, x_k))) over the rows of x_i's class; class radius r_c = max d_i;
- membership ϑ_i = 1 − d_i / (r_c + η);
- kernel distance δ_ij = sqrt(max(0, 2 − 2K(x_i, x_j))); α = max(r_+, r_−); neighbourhood N_i = { j : δ_ij < α } (includes i);
- ρ_i = |{ j ∈ N_i : y_j ≠ y_i }| / |N_i|; non-membership σ_i = (1 − ϑ_i)·ρ_i;
- s_i = ϑ_i if σ_i = 0; 0 if ϑ_i ≤ σ_i; (1 − σ_i)/(2 − ϑ_i − σ_i) otherwise.
`s_min` = scores of X1 rows, `s_maj` = scores of X2 rows, `s_maj_star` = `s_maj` of the X2* rows.

### 5.4 Primal problems (paper eqs. 26–27, typos resolved)
With E = [K(X1, D) e], F* = [K(X2*, D) e], G* = [K(U*, D) e], F2 = [K(X2, D) e], G = [K(U, D) e]:

Plane 1: min ½‖E z1‖² + (C3/2)‖z1‖² + C·s_maj_starᵀξ1 + Cu·eᵀλ1
 s.t. −F* z1 ≥ e − ξ1, G* z1 ≥ (−1 + ε)e − λ1, ξ1 ≥ 0, λ1 ≥ 0.

Plane 2: min ½‖F2 z2‖² + (C3/2)‖z2‖² + C·s_minᵀξ2 + Cu·eᵀλ2
 s.t. E z2 ≥ e − ξ2, ξ2 ≥ 0, λ2 ≥ 0, and the Universum constraint selected by `universum_side2`:
- `"symmetric"` (default; UTSVM eq. 2 of Qi et al.): −G z2 ≥ (−1 + ε)e − λ2;
- `"paper_dual"` (paper eq. 29 and the code): G z2 ≥ (1 − ε)e − λ2;
- `"paper_primal"` (paper eq. 27 as printed): G z2 ≥ (ε − 1)e − λ2.

C plays c1 = c2, C3 plays c3 = c4, Cu is the Universum penalty. The solver formulation may be dual or primal; The corrected-model evaluation compares z1, z2 with the independent evaluation primal solution at relative error ≤ 1e-5.

### 5.5 Decision rule (paper eq. 32)
f_k(x) = |[K(x, D) 1]·z_k| / ‖w_k‖ with ‖w_k‖ the Euclidean norm of z_k without its last entry.
`predict` returns the minority label if f_1(x) < f_2(x), else the majority label, in the caller's original label type.
`decision_function(x)` = (distance to the plane of `classes_[0]`) − (distance to the plane of `classes_[1]`); positive values favour `classes_[1]` (sklearn convention). ROC-AUC uses it.

### 5.6 Estimator API
```python
from ifutsvm import IFUTSVMID
IFUTSVMID(C=1.0, C3=1.0, Cu=1.0, eps=0.3, mu=1.0, eta=1e-4, universum="random_average",
          universum_side2="symmetric", kernel_basis="reduced", random_state=None)
```
sklearn `ClassifierMixin, BaseEstimator`; `fit`, `predict`, `decision_function`, `classes_`. After `fit`, `trace_` is a dict with numpy arrays `X1, X2, X2_star, U, U_star, D, s_min, s_maj, s_maj_star, z1, z2`. Labels may be any hashable values (ints, ±1, strings).

## 6. Pre-specified experiment protocol

- Positive class = minority. Metrics: accuracy, precision, recall (sensitivity), specificity, F1, ROC-AUC (from `decision_function`), G-mean = sqrt(sensitivity·specificity), balanced accuracy, training time (fit wall-clock). `zero_division=0` everywhere.
- **Protocol A** (hold-out): stratified 70/30, seeds {0, 1, 2, 3, 4}; hyper-parameters chosen by stratified 5-fold CV on the 70% only. **Protocol B** (CV): nested stratified 5-fold (outer, seed 0), same inner tuning.
- **Selection metric**: balanced accuracy for the main comparison (all models); accuracy only for the separate reproduction runs of paper-mode IFUTSVM-ID (the paper tuned on accuracy).
- **Search space** (same budget for every model; shrink identically if the compute gate requires): C, Cu, C3 ∈ {10^-3, …, 10^3} (log-uniform), ε ∈ {0.1, 0.3, 0.5, 0.6}, µ ∈ {2^-3, …, 2^3} (log-uniform; on standardised features); SVM γ = 1/(2µ²) over the same µ range. Randomised search with a fixed candidate list per (dataset, split) drawn from a seed shared by all models; the default budget is 40 candidates, documented and identical across models. Parameters a model does not have are ignored, never re-drawn.
- **Permitted compute configuration**: inner tuning CV may be stratified 3-fold (outer protocols unchanged); the main comparison needs ≥ 24 shared candidates per (dataset, split); TSVM/UTSVM may use paper mode's reduced basis D = [X1; X2*] (same X2* draw), documented in `docs/BASELINES.md` with exact-primal tests; complete main benchmark ≤ 6 h wall-clock.
- Features are standardised by a `StandardScaler` fitted inside the pipeline. The faithful model is evaluated as published (no scaling, its own (c, µ, ε) only) **and** with scaling, labelled separately.
- SMOTE: imbalanced-learn `SMOTE(k_neighbors=min(5, n_min−1), random_state=split seed)` inside the pipeline.

