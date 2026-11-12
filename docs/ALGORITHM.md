# Algorithm and two-plane objectives

Symbols follow the corrected specification; the faithful port retains the separate executable
semantics recorded in `PORT_DISCREPANCIES.md`.

## Quantities

Training minority and majority matrices are $X_1$ and $X_2$, with row counts
$m_1\le m_2$. The reduced basis is $D=[X_1;X_2^*]$, where $X_2^*$ samples
$m_1$ majority rows without replacement. The default Universum size is
$u=\max(m_2-m_1,\lceil m_1/2\rceil)$ and the reduced Universum has
$g=\lceil m_1/2\rceil$ rows sampled without replacement from $U$.

$$K(x,z)=\exp\left(-\frac{\|x-z\|^2}{2\mu^2}\right). \tag{1}$$

Augment each kernel matrix with a column of ones:
$E=[K(X_1,D),\mathbf1]$, $F^*=[K(X_2^*,D),\mathbf1]$,
$F_2=[K(X_2,D),\mathbf1]$, $G^*=[K(U^*,D),\mathbf1]$,
$G=[K(U,D),\mathbf1]$. The vectors $s_{\min}$ and $s_{\mathrm{maj}}^*$ are
the full-training-set intuitionistic-fuzzy scores at the corresponding rows.

## Convex plane problems (default symmetric side two)

$$\begin{aligned}
\min_{z_1,\xi_1,\lambda_1}\;&
\frac12\|Ez_1\|^2+\frac{C_3}{2}\|z_1\|^2
+C(s_{\mathrm{maj}}^*)^\top\xi_1+C_u\mathbf1^\top\lambda_1,
\\ \text{subject to }&-F^*z_1\ge\mathbf1-\xi_1,\quad \xi_1,\lambda_1\ge0,
\\ &G^*z_1\ge(\varepsilon-1)\mathbf1-\lambda_1.
\end{aligned} \tag{2}$$

$$\begin{aligned}
\min_{z_2,\xi_2,\lambda_2}\;&
\frac12\|F_2z_2\|^2+\frac{C_3}{2}\|z_2\|^2
+Cs_{\min}^\top\xi_2+C_u\mathbf1^\top\lambda_2,
\\ \text{subject to }&Ez_2\ge\mathbf1-\xi_2,\quad \xi_2,\lambda_2\ge0,
\\ &-Gz_2\ge(\varepsilon-1)\mathbf1-\lambda_2.
\end{aligned} \tag{3}$$

The `paper_dual` alternative replaces the second Universum inequality by
$Gz_2\ge(1-\varepsilon)\mathbf1-\lambda_2$; `paper_primal` uses
$Gz_2\ge(\varepsilon-1)\mathbf1-\lambda_2$. These are distinct supported
formulations. The pre-specified side-two ablation uses `paper_dual`; no sign is
chosen using outer-test performance.

For $z_k=[w_k;b_k]$, predict the minority only when $d_1(x)<d_2(x)$:

$$d_k(x)=\frac{|[K(x,D),1]z_k|}{\|w_k\|_2}. \tag{4}$$

The implementation guards a zero norm with the smallest positive normal
floating-point value. The decision score is oriented to `classes_[1]`; the
evaluation pipeline reorients it to the training-defined minority for ROC-AUC.

## Algorithm 1 (corrected implementation)

```text
Input: one training partition, labels, fixed seed and candidate parameters
A. Fit nominal encoding and scaling using this partition only.
B. Determine minority/majority roles; create the local seeded generator.
C. Draw X2-star without replacement; generate U from training data only.
D. Draw U-star without replacement; assemble the reduced kernel basis D.
E. Compute kernel distances, class radii, memberships, local opposite-class
   proportions, non-memberships and fuzzy weights on the full training partition.
F. Assemble E, F-star, F2, G-star and G; solve the two convex plane problems.
G. Return the fitted preprocessing steps, basis, planes and construction trace.
Prediction: transform new rows with the fitted preprocessor, calculate both
plane distances, choose the closer plane, and expose continuous decision scores.
```

Line groups A-F are repeated independently inside each inner training fold.
After training-only selection, A-G are refitted on the full outer training
partition, followed by one outer evaluation. Exact objectives and sampler
contracts are checked against independent primals and moment/support checks.
The minority-generative conditional law and the limits of its theory are in
`THEORY.md`; baseline objectives are in `BASELINES.md`.
## Numerical stability

The corrected kernel estimates squared-distance roundoff using
`8 * machine_epsilon * (features + 2) * (squared_norm_x + squared_norm_y)`.
The factor is a conservative allowance for summation and dot-product error.
A mean-value bound propagates that allowance through the exponential, using
the lowest possible nonnegative distance. If the estimated absolute kernel
error exceeds the fixed `1e-9` accuracy budget, that pair uses SciPy's direct
squared differences. Other pairs retain the original arithmetic and order.
This numerical criterion uses feature values and bandwidth only.

Run `python -m pytest tests/test_kernel_translation.py -q` for large-translation,
direct-distance, fitted-model and empty-matrix checks.

