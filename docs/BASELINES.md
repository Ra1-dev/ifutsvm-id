# Baseline contracts

References: Jayadeva, Khemchandani and Chandra, IEEE TPAMI 29(5), 2007 (TSVM);
Qi, Tian and Shi, Neural Networks 36, 2012 (UTSVM). Universum signs follow the
UTSVM problems restated in the main IFUTSVM-ID paper, equations 1 and 2.

The main comparison uses the **reduced** basis D=[X1;X2*],
with exactly paper mode's seeded majority-subset draw. The full D=[X1;X2]
variant remains available and independently tested; the superseded full-basis
benchmark is a pilot. Let E=[K(X1,D),1],
F=[K(X2,D),1], G=[K(U,D),1], with the same Gaussian bandwidth convention as
paper mode. X1 is the training minority. All slack variables are nonnegative.

TSVM plane 1 minimizes `0.5||E z1||² + delta/2 ||z1||² + C sum(xi1)` subject
to `-F z1 >= 1-xi1`. Plane 2 minimizes
`0.5||F z2||² + delta/2 ||z2||² + C sum(xi2)` subject to `E z2 >= 1-xi2`.

UTSVM adds `Cu sum(lambda)` to each objective and the constraints
`G z1 >= eps-1-lambda1` and `-G z2 >= eps-1-lambda2`. Both planes use all U.
Universum samples equal paper-mode random averages for the same training rows
and seed. E and F still contain all minority and majority rows respectively;
only the coefficient expansion basis is reduced.

The classic TSVM/UTSVM objectives lack structural regularization. This
implementation explicitly includes a fixed numerical ridge `delta=1e-5` on
coefficients **and bias** to make singular full-basis problems unique. Results
are labeled numerically regularized TSVM/UTSVM; no claim of exact unregularized
coefficients is made. Delta is fixed, not tuned as C3. The independent cvxpy
tests solve these exact stated primals and compare coefficients at relative
error <=1e-5.

Distances use the coefficient norm as in the published kernel twin-SVM rule.
Labels and decision-function orientation follow the sklearn contract.

SVM uses an RBF kernel with balanced class weights and gamma=1/(2 mu²).
SMOTE+SVM applies SMOTE only inside each training pipeline, with neighbour
count bounded by the training minority size. All models share outer/inner
indices and candidate draws. Paper mode uses the symmetric Universum side
fixed before results are inspected. Faithful mode retains the demo's errors,
kernel scales and label defect, with raw and scaled variants labeled separately.
