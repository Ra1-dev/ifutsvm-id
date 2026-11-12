# Conditional Gaussian Universum: working proposition

This is a research specification and factual derivation. It is not a claim of
improved benchmark performance. The pre-specified research protocol specifies the empirical requirements; the initial transfer prediction
was committed before its synthetic outcomes.

## Construction and objective

Fit class means and a shared covariance to the training fold. The covariance
is the equally weighted mean of separate Ledoit–Wolf class covariance estimates,
plus the declared numerical ridge. Equal class weights are deliberate: the
majority count does not determine their mixture weight. Denote the fitted means
by mu1 and mu2, covariance by Sigma, delta=mu2-mu1, midpoint m=(mu1+mu2)/2,
a=Sigma^{-1}delta and d²=deltaᵀSigma^{-1}delta.

The objective is to sample the fitted **minority Gaussian conditional on equal
class-conditional density**, rather than to duplicate labeled minority samples.

The single sampling objective is

`Q_U = Law(X | a^T(X-m)=0), X ~ N(mu1,Sigma)`.

This is the fitted minority density restricted to the density-neutral region,
where an unlabeled constraint can oppose prior-induced boundary displacement.
There are zero tuned strategy hyperparameters. The relative covariance ridge
is a fixed numerical constant (1e-8), not selected on outcomes.

The conditioning event is aᵀ(x-m)=0. It ignores empirical class priors and is
defined by training features only. The resulting Gaussian has mean m and
covariance Sigma-delta deltaᵀ/d². It is supported on the density-neutral
hyperplane. When the fitted means coincide, the conditioning adds no restriction
and the shared Gaussian is used. In one dimension the nondegenerate conditional
law is a point mass at the midpoint; repeated points are intentional.

Implementation: `ifutsvm.universum.ConditionalGaussianUniversum`, usable as
`IFUTSVMID(universum=ConditionalGaussianUniversum(), random_state=seed)`.
It returns the requested number of points using the supplied RNG and never sees
validation labels, test features, class prior targets or cached fixtures.
For very small nonzero mean separation, the implementation rescales its
direction before forming the outer product, preserving the conditional law
when unscaled products would underflow. This numerical branch does not change
the model objective. Sampling support, moments and deterministic behavior are
checked in `tests/test_generative.py`.

## Proposition and proof

Assume two population distributions N(mu1,Sigma) and N(mu2,Sigma), Sigma positive
definite and distinct means. Then:

1. Every generated point has equal population class-conditional density.
2. Its sampling distribution is independent of the class-count ratio r.
3. A neutral-boundary anchor can reduce the imbalance-induced boundary shift
   in the explicitly stated threshold surrogate below.

For (1), expanding the two quadratic log densities gives
`log(p2(x)/p1(x))=aᵀ(x-m)`. Gaussian conditioning of X~N(mu1,Sigma) on
`aᵀX=aᵀm` gives mean
`mu1+Sigma a (aᵀm-aᵀmu1)/(aᵀSigma a)=m` and covariance
`Sigma-Sigma a aᵀSigma/(aᵀSigma a)=Sigma-delta deltaᵀ/d²`.
This proves the support and density claim. No r occurs, proving (2).

For (3), use the standardized separation coordinate
`s=aᵀ(x-m)/d`. It is N(-d/2,1) in class 1 and N(d/2,1) in class 2.
The ordinary prior-weighted Gaussian decision threshold is
`b=-log(r)/d`; the balanced-error optimal threshold is zero. Consider the
explicit surrogate `alpha*(t-b)^2 + lambda*t^2`, alpha>0, lambda>0. The second
term anchors the boundary to the generated density-neutral support. Its unique
solution is `t=alpha*b/(alpha+lambda)`, closer to zero than b for r != 1.
Balanced error is
`R(t)=0.5*(1-Phi(t+d/2)+Phi(t-d/2))`.
Its derivative is `0.5*(phi(t-d/2)-phi(t+d/2))`, negative for t<0 and
positive for t>0. Therefore anchoring strictly lowers balanced error when r != 1,
and leaves it unchanged at r=1. This is the precise imbalance-dependent result.

## Limits and falsification

The theorem is about the population construction and the stated threshold
surrogate. It is **not** a proof that IFUTSVM-ID's asymmetric hinge constraints
implement that quadratic anchor or improve its predictions. The actual model
already uses several imbalance corrections. Universum constraints can be
inactive, estimated class means can be noisy, Gaussian classes can be wrong,
and the generated support can overlap real classes. Equal density means
uncertain class identity; it does not prove a third semantic class exists.

The analytical check must compare the derived conditional moments and density
ratio with generated samples, and measure threshold risk across count ratios.
The actual-estimator check must train on fixed overlapping Gaussian problems
with varying imbalance, compare random averaging and conditional generation,
and retain every seed and penalty. It must report whether actual decision
boundaries move toward the balanced boundary and whether balanced error improves.
Failure of either transfer prediction is a negative research result, not a
reason to choose a favorable seed, discard a penalty or weaken the proposition.

The KEEL ablation compares no Universum, code construction, random averaging,
noise and this construction on the same splits, plus size and fuzzy-score
controls. Those outcomes, rather than this toy surrogate alone, decide whether
the proposed strategy addresses the practical gap.

## Pre-specified outcome

The quantitative transfer criterion is recorded in
`results/research/theory_plan.json`. The original 336-run outcome remains in
`results/research/theory_empirical.json`. Fresh per-example predictions follow
from `python -m experiments.theory_check --output tmp/theory-rerun`.
The criterion is refuted: the mean
boundary movement has the opposite sign. This does not falsify the population
Gaussian-conditioning identity or establish an IFUTSVM performance guarantee.
The complete KEEL contrasts are recorded below.

## What the proposition does not distinguish

Under the same population shared-Gaussian assumptions, random averaging also
has midpoint mean. In the scalar quadratic threshold surrogate, its variance
adds a constant to the expected anchor loss, so it produces the same optimal
threshold as the conditional law. The proposition therefore predicts benefit
versus an unanchored prior threshold, not superiority over random averaging.
The pre-specified actual-estimator contrast against random averaging was an
additional empirical transfer hypothesis; its refutation is retained. The
conditional law has zero variance normal to the neutral hyperplane, whereas
random averaging has variance one half in the standardized separation
coordinate. This is a construction distinction, not a classification guarantee.
Finite-sample fitted distributions also depend on class sample counts through
estimation error; prior-independence in the proposition assumes population
parameters are known.

<!-- EMPIRICAL OUTCOME -->
## Recorded empirical outcome

- Original pre-specified transfer criterion: refuted; 336 retained synthetic records.
- High-IR balanced-accuracy change: +0.015277778 percentage points.
- Absolute-boundary-distance change: +0.0175812479949; the pre-specified criterion requires movement toward the balanced boundary.
- Synthetic numerical source: `results/research/theory_empirical.json`; all original contrasts and criteria: `results/research/theory_plan.json`.
- Original pre-registration SHA-256: `ff156fe4222f35ed88f8550853e3d6e812976ad5acd868cdc241e870032a251c`.
- Complete KEEL ablation: 900 outer cells over 12 datasets.
- Comparison below: IFUTSVM-ID (conditional Gaussian) minus IFUTSVM-ID (corrected); dataset means over the fixed seeds.
- KEEL contrasts are external-validity evidence, not a replacement prediction or a proof that the threshold surrogate transfers to the estimator.
- Signed-rank tests assume independent, symmetric paired differences; source-family sensitivity and every declared contrast remain in `docs/RESEARCH.md` and `results/research/statistics.json`.

| Metric | Mean change pp | Median change pp | W | Raw p | Holm p | Significant | Source |
|---|---:|---:|---:|---:|---:|---|---|
| G-mean | +0.680558 | +0.432483 | 14 | 0.359375 | 1 | False | `results/research/statistics.json#/paired/comparisons/0` |
| Balanced accuracy | +0.525683 | +0.486644 | 14 | 0.359375 | 1 | False | `results/research/statistics.json#/paired/comparisons/3` |
| F1 | -0.185079 | +0.000000 | 20 | 0.8203125 | 1 | False | `results/research/statistics.json#/paired/comparisons/6` |
| ROC-AUC | +1.032406 | +0.339961 | 9 | 0.064453125 | 0.7734375 | False | `results/research/statistics.json#/paired/comparisons/9` |

<!-- END EMPIRICAL OUTCOME -->
