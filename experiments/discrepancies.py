"""Deterministic, single-intervention evidence for the discrepancy register.

Run from repository root. No evaluation data are used for parameter selection.
These are diagnostic synthetic cases, not benchmark performance claims.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from ifutsvm import IFUTSVMID
from ifutsvm._numeric import aug, rbf, scores, hinge_plane
from ifutsvm.reference import trace_reference, octave_indices, evaluate, reference_prediction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    rng = np.random.default_rng(20261109)
    X = np.vstack((rng.normal(.35, 1, (18, 4)), rng.normal(-.15, 1.2, (61, 4))))
    y = np.r_[np.ones(18), -np.ones(61)]
    Xt = np.vstack((rng.normal(.35, 1, (17, 4)), rng.normal(-.15, 1.2, (43, 4))))
    yt = np.r_[np.ones(17), -np.ones(43)]
    prm = dict(C=.7, C3=.4, Cu=.2, mu=1.7, eps=.3, random_state=91)
    model = IFUTSVMID(**prm).fit(X, y)
    tr = model.trace_
    ref = trace_reference(np.column_stack((X, y)), np.column_stack((Xt, yt)), .7, 1.7, .3)
    rows = []

    def add(where, paper, code, executable, faithful, corrected, metric, value):
        rows.append(dict(id=f"DSC-{len(rows)+1:02}", where=where, paper=paper,
                         code=code, executable=executable, faithful=faithful,
                         corrected=corrected, metric=metric,
                         value=None if value is None else float(value)))

    def solve(zero_kernel_scale=False, ridge=.4, Cu=.2, major=None, U=None, Us=None, score_mu=None, side2='symmetric'):
        A, B = tr['X1'], tr['X2']
        Bs = tr['X2_star'] if major is None else major
        U = tr['U'] if U is None else U
        Us = tr['U_star'] if Us is None else Us
        D = np.vstack((A, Bs))
        mu = np.sqrt(prm['mu']) if zero_kernel_scale else prm['mu']
        E, F, G, Fs, Gs = [aug(rbf(a, D, mu)) for a in (A, B, U, Bs, Us)]
        s = scores(np.vstack((A, B)), np.r_[np.ones(len(A)), np.zeros(len(B))], score_mu or prm['mu'])
        idx = np.argmin(((Bs[:, None, :] - B[None, :, :])**2).sum(2), axis=1)
        z1 = hinge_plane(E, np.vstack((-Fs, Gs)), np.r_[np.ones(len(Bs)), np.full(len(Us), -.7)],
                         np.r_[.7*s[len(A):][idx], np.full(len(Us), Cu)], ridge)
        sign, rhs = (-1, -.7) if side2 == 'symmetric' else (1, .7)
        z2 = hinge_plane(F, np.vstack((E, sign*G)), np.r_[np.ones(len(A)), np.full(len(U), rhs)],
                         np.r_[.7*s[:len(A)], np.full(len(U), Cu)], ridge)
        return np.r_[z1, z2]

    base = solve()
    delta = lambda z: np.linalg.norm(z-base)
    add('kernelfun.m:41-48; supplementary S.III-A', 'RBF denominator 2 mu^2', 'training denominator 2 mu',
        'Training/test bandwidths disagree except mu=1', 'retain', 'one RBF everywhere',
        'plane L2 delta changing only training bandwidth', delta(solve(zero_kernel_scale=True)))
    add('non_linear_score_values.m:7-9; supplementary S.III-A', 'same Gaussian kernel', 'score denominator mu^2',
        'Score bandwidth differs from testing by sqrt(2)', 'retain', 'same kernel for scores and planes',
        'plane L2 delta changing only score bandwidth', delta(solve(score_mu=prm['mu']/np.sqrt(2))))
    add('IFUTSVM.m:39; paper reduced X2* construction', 'random subset', 'deterministic spaced row indices',
        'Order-dependent subset', 'retain', 'sample without replacement',
        'plane L2 delta replacing X2* by code subset', delta(solve(major=ref['B1'])))
    add('main.m:47-57; supplementary S.III-A', 'random averaging', 'spaced rows and prefix padding',
        'Repeated indices possible; no random draw', 'retain', 'independent opposite-class draws',
        'plane L2 delta replacing U by code U under paper_dual side',
        np.linalg.norm(solve(U=ref['U'],side2='paper_dual')-solve(side2='paper_dual')))
    iu = octave_indices(1, (len(tr['U'])-1)/9, len(tr['U'])-(len(tr['U'])-1)/9, len(tr['U']))
    add('IFUTSVM.m:61-62,76; Algorithm 1 step 1', 'random U* subset', 'spaced U rows',
        'No random draw; order-dependent', 'retain', 'sample U without replacement',
        'plane L2 delta replacing only U* by spaced subset', delta(solve(Us=tr['U'][iu])))
    idx = octave_indices(1, 1.5, 7, 7)
    add('main.m:49-50; IFUTSVM.m:23,39,76', 'sample valid rows', 'fractional colon indices',
        'Octave rounds halves away from zero; MATLAB rejects fractional indices', 'Octave rounding', 'integer RNG indices',
        'fractional indices rounded in five-element probe', np.count_nonzero(idx+1 != np.arange(1, 7.1, 1.5)))
    errors = 0
    for counts in ((4,4),(4,5)):
        a = rng.normal(size=(sum(counts), 2)); b = np.r_[np.ones(counts[0]), -np.ones(counts[1])]
        try:
            trace_reference(np.column_stack((a,b)), np.column_stack((a,b)), 1, 1, .3)
        except (ValueError, IndexError):
            errors += 1
    add('IFUTSVM.m:23,61; paper u=m2-m1', 'no near-balance guard specified', 'zero/undefined sampling increment',
        'Balanced and difference-one examples raise', 'raises', 'U count floor keeps construction defined',
        'unsupported count probes that raise', errors)
    near = IFUTSVMID(random_state=0).fit(X[:37], np.r_[np.ones(18), -np.ones(19)])
    add('paper U and U* sizes; docs/SPECIFICATION.md 5.2', 'u=m2-m1; g=ceil(m1/2)', 'may request more U* than available distinct rows',
        'Fractional indices can repeat rows', 'retain where executable', 'u=max(m2-m1,g), deliberate extension',
        'extra U rows added by floor on 18/19 probe', len(near.trace_['U'])-1)
    add('IFUTSVM.m:27,74,80; paper primal regularizers', 'tunable c3=c4 on weights and bias', 'fixed ep=1e-5 in both inverses',
        'c3 variable instead bounds Universum slack', 'retain ep', 'C3 independently controls ridge',
        'plane L2 delta replacing C3 by demo ridge', delta(solve(ridge=1e-5)))
    add('IFUTSVM.m:25-26,69-70; supplementary S.III-A', 'Cu independently tuned from classification C', 'all slack penalties tied to c1',
        'Cannot tune Universum strength independently', 'retain', 'independent C and Cu',
        'plane L2 delta tying Cu to C', delta(solve(Cu=.7)))
    dual = IFUTSVMID(**prm, universum_side2='paper_dual').fit(X,y)
    printed = IFUTSVMID(**prm, universum_side2='paper_primal').fit(X,y)
    add('paper eqs.27/29 (case numbering); IFUTSVM.m:86,93', 'printed primal Gz >= eps-1; dual implies Gz >= 1-eps; UTSVM symmetric uses -Gz >= eps-1',
        'plus O contribution and 1-eps dual coefficient', 'Executes paper_dual sign', 'paper_dual', 'three explicit variants; symmetric default',
        'z2 L2 difference dual versus printed-primal variants', np.linalg.norm(dual.trace_['z2']-printed.trace_['z2']))
    swapped = trace_reference(np.column_stack((X,-y)), np.column_stack((Xt,-yt)), .7, 1.7, .3)
    add('IFUTSVM.m:10-18,108-113', 'plane class identity preserved', 'swaps data and scores but outputs +1 for plane 1',
        'When +1 majority, minority plane is mislabeled', 'retain', 'map planes to original labels',
        'predictions changed by correcting swapped-label mapping', np.count_nonzero(swapped['classifier'] != -swapped['classifier']))
    ev = evaluate(yt, ref['classifier'])
    add('IFUTSVM.m:148-150; supplementary eq.7', 'accuracy=(TP+TN)/N', 'first return is 100 times balanced accuracy',
        'Returned accuracy differs from Evaluate accuracy', 'retain both', 'caller metrics use actual accuracy',
        'absolute percentage-point difference on overlapping probe', abs(float(ref['acc_returned'][0])-100*ev['eval_accuracy']))
    pred = model.predict(Xt); evp = evaluate(yt, pred)
    add('Evaluate.m:37; course ROC-AUC requirement', 'paper does not report ROC-AUC; course requires score AUC',
        'AUC uses hard labels: mean sensitivity and specificity', 'One-threshold AUC, not ranking AUC', 'retain', 'decision_function supports ranking AUC',
        'absolute AUC difference hard labels versus continuous scores', abs(evp['eval_AUC']-roc_auc_score(yt, model.decision_function(Xt))))
    add('Evaluate.m:44; docs/SPECIFICATION.md 6', 'conventional G-mean=sqrt(sensitivity*specificity)', 'sqrt(precision*recall)',
        'Depends on prevalence through precision', 'retain', 'experiments use conventional G-mean',
        'absolute G-mean difference on overlapping probe', abs(evp['eval_gmean']-np.sqrt(evp['eval_sensitivity']*evp['eval_specificity'])))
    zero = evaluate(yt, -np.ones(len(yt)))
    add('Evaluate.m:41-44', 'zero-division policy not specified', 'precision and F1 have no zero guard',
        'All-negative predictions produce NaN precision/F1/gmean', 'retain NaN', 'experiment metrics will declare zero_division policy',
        'nonfinite metric count for all-negative predictions', sum(not np.isfinite(v) for v in zero.values()))
    add('non_linear_score_values.m:19-20; supplementary eq.1', 'eta positive parameter', 'eta fixed at 1e-4',
        'Cannot vary radius smoothing', 'retain', 'eta exposed',
        'max score change eta 1e-4 versus .1', np.max(np.abs(scores(X,y,1.7)-scores(X,y,1.7,.1))))
    K = rbf(X,X,1.7); distance = np.sqrt(np.maximum(0,2-2*K))
    radii = []
    for label in (-1,1):
        kc = K[np.ix_(y==label,y==label)]
        radii.append(np.sqrt(np.maximum(0,1-2*kc.mean(1)+kc.mean())).max())
    alpha = max(radii)
    add('supplementary eq.5; non_linear_score_values.m:17,26', 'numerator radius alpha but denominator eta as printed',
        'same radius alpha=max(class radii) for both', 'Avoids mismatched neighbour populations', 'retain code', 'retain coherent same-radius interpretation',
        'neighbour entries differing between alpha and eta=1e-4', np.count_nonzero((distance<alpha)!=(distance<=1e-4)))
    add('supplementary eq.5; non_linear_score_values.m:26', 'closed neighbourhood <=', 'strict <',
        'Boundary points excluded; zero-radius neighbourhood empty', 'strict', 'strict with self fallback only at degeneracy',
        'boundary neighbours excluded in distances [0,.5,1] radius .5', np.count_nonzero((np.array([0,.5,1])<=.5)!=(np.array([0,.5,1])<.5)))
    add('main.m:59-61; supplementary S.III-A', 'grid search over C,Cu,eps,mu with five-fold CV',
        'one fixed triple (1,1,1), no grid or CV loop', 'eps=1 outside listed epsilon grid', 'API accepts supplied parameters', 'API tunable; experiment protocol must nest tuning',
        'distance from demo eps to closest published epsilon', min(abs(1-e) for e in (.1,.3,.5,.6)))
    add('abstract/conclusion oversampling claim; IFUTSVM.m:39', 'claims oversampling and undersampling',
        'reduces majority subset; creates unlabeled Universum only', 'No additional labeled minority observations', 'retain', 'no claim of labeled oversampling',
        'added labeled minority rows', len(tr['X1'])-np.count_nonzero(y==1))
    add('supplementary S.II complexity; non_linear_score_values.m:7-9,23-26', 'linear score-computation work claimed',
        'dense full pairwise kernel and neighbourhood scan', 'Quadratic kernel storage and distance work', 'retain dense scoring', 'dense scoring documented; no linear complexity claim',
        'full score kernel entries / training rows', len(X)**2/len(X))
    add('supplementary Algorithm 1 input; main.m:40', 'prints u=m2-m2 (typo); main text says m2-m1',
        'absolute difference in class counts', 'Nonzero U despite printed zero', 'retain difference', 'difference with near-balance floor',
        'U count difference from literal Algorithm 1 typo', len(tr['U']))
    add('IFUTSVM.m:88-89; docs/SPECIFICATION.md 2', 'solve convex QP', 'quadprog with nearly singular PSD dual Hessian',
        'Independent reference evaluation records Octave rejection/zeros; MATLAB oracle solves box QP', 'MATLAB-semantic oracle as specified', 'strongly convex slack primal',
        'z2 norm lost if solver substitutes zero dual vector', np.linalg.norm(ref['w2']))
    zero_result=reference_prediction(Xt,ref['C'],ref['w1'],np.zeros_like(ref['w2']),1.7)
    rows[-1]['executable']+='; zero w2 makes y2=0/0 NaN, so every comparison is false and every prediction is -1 (majority for the normal orientation)'
    rows[-1]['additional_measurements']=dict(nan_y2_count=int(np.isnan(zero_result[2]).sum()),
                                            majority_predictions=int(np.sum(zero_result[3]==-1)))
    add('IFUTSVM.m:23; main.m:57; paper U construction', 'constructed U used as the majority-plane Universum',
        'resamples incoming U again with step (ux-1)/d',
        'For ux=d>2, rounds to a duplicate interior index and omits the final row',
        'retain second-stage U', 'use generated U directly; only U* is subsampled',
        'maximum coordinate change main_U versus internal U', np.max(np.abs(ref['main_U']-ref['U'])))
    add('IFUTSVM.m:4,23-46,90-106; supplementary Algorithm 1',
        'algorithm includes score construction and plane recovery; timing scope not specified',
        'timer starts after scores and sampling and ends before plane recovery and testing',
        'returned time excludes required fit work, not an end-to-end fit duration',
        'trace API does not expose timer; external wall-clock includes whole call',
        'benchmark measures entire fit',
        'score-kernel entries excluded before tic',len(X)**2+18**2+61**2)
    add('supplementary S.II complexity comparison',
        'claims IR^3*m1^3+u^3 is much smaller than m1^3+m2^3+u^3 when m1 is small',
        'two QPs of dimensions m1+ceil(m1/2) and m1+u; full dense score kernel',
        'IR^3*m1^3 equals m2^3 exactly; substituting IR alone proves no asymptotic speedup',
        'retains actual problem dimensions', 'reports measured timing instead of the claimed algebraic speedup',
        'ratio of IR-substituted majority term to original majority term',(61/18)**3*18**3/61**3)
    E=aug(rbf(tr['X1'],tr['D'],prm['mu']))
    G=aug(rbf(tr['U'],tr['D'],prm['mu']))
    wrong_F=hinge_plane(E,np.vstack((E,-G)),np.r_[np.ones(len(E)),np.full(len(G),-.7)],
                        np.r_[.7*tr['s_min'],np.full(len(G),.2)],.4)
    add('paper dual definitions following linear/nonlinear duals; IFUTSVM.m:79-82',
        'F is printed using X1, although the majority-plane primal uses X2',
        'inverse for plane 2 uses the majority design G',
        'Literal printed F changes the Hessian; the primal requires majority rows',
        'majority Hessian', 'derive Hessian from majority primal, resolve F typo',
        'z2 L2 delta substituting the printed minority F',np.linalg.norm(wrong_F-tr['z2']))
    count_deltas=[]
    for split_seed in range(5):
        _,unstratified=train_test_split(np.arange(len(y)),test_size=.3,random_state=split_seed)
        _,stratified=train_test_split(np.arange(len(y)),test_size=.3,random_state=split_seed,stratify=y)
        count_deltas.append(abs(np.sum(y[unstratified]==1)-np.sum(y[stratified]==1)))
    add('main.m:24-29; supplementary S.III-A',
        'one random 70/30 split; no repeats or variance supplied',
        'cvpartition receives row count rather than labels; rng default resets for each dataset',
        'Non-stratified single holdout; five-fold tuning described in text is absent',
        'trace takes pre-split inputs; no invented MATLAB partition equivalence',
        'external runner uses stratification and repeated holdouts/nested CV',
        'sum of minority test-count changes over five fixed policy probes',sum(count_deltas))
    full=IFUTSVMID(**prm,kernel_basis='full').fit(X,y)
    add('paper nonlinear basis definition D=[X1;X2]; IFUTSVM.m:42',
        'nonlinear equations define a full basis, despite reduced-kernel description',
        'basis is [A;B1] with only sampled majority rows',
        'Coefficient count and kernel span are reduced',
        'reduced basis', 'explicit reduced default and full variant',
        'test-score L2 delta full versus reduced basis',np.linalg.norm(full.decision_function(Xt)-model.decision_function(Xt)))
    KD=rbf(tr['D'],tr['D'],prm['mu']); KT=aug(rbf(Xt,tr['D'],prm['mu']))
    distances=[]
    for key in ('z1','z2'):
        w=tr[key][:-1]
        distances.append(np.abs(KT@tr[key])/np.sqrt(w@KD@w))
    add('paper kernel decision equation; IFUTSVM.m:104-105',
        'printed coefficient Euclidean norm; a geometric RKHS distance would use sqrt(w^T K(D,D) w)',
        'Euclidean norm of expansion coefficients, not RKHS norm',
        'Normalisation depends on coefficient representation; not an RKHS distance',
        'literal coefficient norm', 'literal coefficient norm per docs/SPECIFICATION.md; RKHS alternative diagnostic only',
        'test-score L2 delta replacing Euclidean with RKHS norm',np.linalg.norm((distances[1]-distances[0])-model.decision_function(Xt)))
    # These values are factual transcriptions from the supplied supplementary
    # tables, not experiment results. Their arithmetic audit is generated here.
    table_a=np.array([690,14,86.47,.0001,.1,.3,2])
    table_b=np.array([690,14,86.47,.0001,.1,.3,2])
    add('supplementary Table S.II aus/checkerboard Data; Table S.I',
        'different dataset names but identical size, IFUTSVM accuracy and published parameters',
        'demo provides no dataset identity manifest to resolve the ambiguity',
        'Reported rows cannot establish that distinct data were evaluated',
        'does not infer identity', 'downloads explicit datasets with hashes; flags ambiguity',
        'identical published fields in the two Table S.II entries',np.sum(table_a==table_b))
    add('supplementary Algorithm 1 steps 3-4; linear/nonlinear sections',
        'Algorithm references plane/decision equation numbers inconsistent with supplied main-text versions',
        'executable code uses defined matrices, not equation-number lookup',
        'Algorithm nonlinear references 36,37,38 exceed HTML-v2 final model equation 34',
        'executable formulas', 'explicit docs/SPECIFICATION.md primals; version-qualified citations',
        'nonlinear Algorithm 1 references beyond model equation range',sum(k>34 for k in (36,37,38)))
    add('main text linear/nonlinear notation and supplementary Table S.II',
        'eta is a smoothing parameter and a dual vector; sigma denotes bandwidth in tables but nonmembership elsewhere; C3 is not tabulated',
        'mew/mu denotes bandwidth; score smoothing and ridge are fixed',
        'Four tabulated hyperparameters do not identify independently tunable ridge',
        'fixed ridge', 'separate mu,eta,C3; published-config ridge default declared explicitly',
        'independent paper-fit hyperparameters absent from four-column table',len(('C','C3','Cu','eps','mu'))-len(('C','Cu','eps','sigma')))
    add('main text inversion-bypass claim; IFUTSVM.m:74,80',
        'claims bypassing matrix inversion', 'explicitly forms two inverse matrices',
        'Reduced dimensions do not eliminate inversion', 'two explicit inverses',
        'Cholesky factor and triangular solves, no explicit inverse in paper solver',
        'explicit inverses required by a faithful fit',len(('invHTH','invGTG')))
    add('main text block stationarity equation (HTML-v2 eq.21); IFUTSVM.m:92',
        'the U* block is printed without its beta multiplier, unlike adjacent KKT/plane equations',
        'multiplies the Universum design transpose by its dual vector',
        'Literal missing multiplier leaves a matrix instead of the required coefficient vector',
        'retains multiplier', 'derive stationarity from the primal; retain multiplier',
        'excess columns of the unmultiplied Universum block',len(tr['U_star'])-1)
    w=tr['z1'][:-1]
    add('main text nonlinear primal regularizer (HTML-v2 eqs.28-29)',
        'prints norm(w squared), while linear primal and dual imply squared norm(w)',
        'quadratic identity ridge', 'Literal norm of componentwise squares is not the quadratic ridge',
        'quadratic ridge', 'quadratic ridge per docs/SPECIFICATION.md',
        'difference between squared norm and norm of componentwise squares',abs(np.linalg.norm(w)**2-np.linalg.norm(w**2)))
    data_manifest=Path('data/keel/MANIFEST.json')
    if data_manifest.exists():
        manifest=json.loads(data_manifest.read_text())['datasets']
        published=json.loads(Path('data/published_table_sii.json').read_text())['datasets']
        add('supplementary Table S.II abalone9-18; official KEEL header',
            'lists seven features', 'demo iterates whichever feature columns are supplied',
            'official full data includes seven numeric features plus nominal Sex; numeric-only data would differ',
            'uses supplied numeric design', 'loader preserves all published columns; categorical encoding inside training pipeline',
            'official minus Table S.II feature count',manifest['abalone9-18']['features']-published['abalone9-18']['features'])
        glass=published['glass4']
        test_sizes={int(np.floor(.3*glass['rows'])),int(np.ceil(.3*glass['rows']))}
        discrepancy=min(abs(100*round(glass['accuracy_percent']*n/100)/n-glass['accuracy_percent'])
                        for n in test_sizes)
        add('supplementary Table S.II glass4; single 70/30 evaluation description',
            'reports ordinary ACC for one random holdout, but its value is off the integer-correct-count lattice',
            'returned acc is balanced accuracy, while Evaluate also computes ordinary accuracy',
            'For either floor or ceil test size, the reported value cannot be rounded ordinary single-split accuracy; an unstated aggregation or metric convention is needed',
            'preserves returned metric distinction', 'stores confusion counts and declares aggregation; does not infer the authors actual protocol',
            'minimum percentage-point distance to feasible glass4 holdout accuracy',discrepancy)
    row_zscore=lambda a:(a-a.mean(axis=1,keepdims=True))/a.std(axis=1,ddof=1,keepdims=True)
    row_model=IFUTSVMID(**prm).fit(row_zscore(X),y)
    scaler=StandardScaler().fit(X)
    feature_model=IFUTSVMID(**prm).fit(scaler.transform(X),y)
    add('main.m:12-15 (commented line 13)',
        'no complete feature-scaling procedure is specified',
        'zscore of the transposed feature matrix is commented out; demo uses raw feature scales',
        'Enabling the comment standardizes each row across features with sample std, not each feature across training rows',
        'raw supplied design; faithful_raw benchmark',
        'train-fold feature StandardScaler; rowwise alternative diagnostic only',
        'test-score L2 delta raw versus enabled rowwise comment',
        np.linalg.norm(row_model.decision_function(row_zscore(Xt))-model.decision_function(Xt)))
    rows[-1]['additional_measurements']=dict(
        feature_standardization_score_L2=float(np.linalg.norm(feature_model.decision_function(scaler.transform(Xt))-model.decision_function(Xt))),
        rowwise_prediction_changes=int(np.sum(row_model.predict(row_zscore(Xt))!=pred)),
        featurewise_prediction_changes=int(np.sum(feature_model.predict(scaler.transform(Xt))!=pred)))
    add('HTML-v2 III-A after eq.11 and III-B after eq.29; IFUTSVM.m:5-18; non_linear_score_values.m outputs',
        'linear description lists negative:s2 then positive:s1; nonlinear description lists positive:s1 then negative:s2 (order reverses, label mapping agrees)',
        'initial s1/s2 refer to +1/non-+1 labels; swap branch exchanges both data and score vectors when +1 is larger',
        'After the branch s1 belongs to minority A and s2 to majority B, regardless of original label; never infer class identity from a bare score-vector name',
        'preserve the reassigned s1/s2 values',
        'explicit s_min, s_maj, s_maj_star tied to class rows',
        'notation-only clarification; no isolated performance effect claimed',None)
    rows[-1]['source']='https://arxiv.org/html/2410.20335v2#S3'
    output = dict(seed=20261109, parameters=prm,
                  probe_accuracy=float(np.mean(pred==yt)), rows=rows)
    path = Path('results/discrepancies.json'); path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8', newline='\n')
    text = ['# Port discrepancy register', '',
            'Generated by `python experiments/discrepancies.py` from `results/discrepancies.json`.',
            'Source locations are in the ignored authors\' reference; paper equation numbering follows the supplied paper notes (HTML v2 renumbers equations).',
            'Independent overlapping synthetic probe; impacts are interventions or diagnostic counts, not benchmark gains. No test tuning.',
            'The Octave solver rejection is independent reference-evaluation evidence; DSC-24 measures its zero-solution consequence, not a local Octave run.', '',
            '| ID | where | paper says | code does | executable behaviour (Octave/MATLAB) | faithful mode | paper mode | measured impact (command + number) |',
            '|---|---|---|---|---|---|---|---|']
    for row in rows:
        cells = [row[k] for k in ('id','where','paper','code','executable','faithful','corrected')]
        measured='not applicable' if row['value'] is None else f"{row['value']:.10g}"
        cells.append(f"`python experiments/discrepancies.py`: {row['metric']} = {measured}")
        if 'source' in row:
            cells[1]+=f"; [paper source]({row['source']})"
        if 'additional_measurements' in row:
            cells[-1]+='; '+json.dumps(row['additional_measurements'],sort_keys=True)
        text.append('| '+' | '.join(cells)+' |')
        print(f"{row['id']}: {row['metric']} = {measured}")
        if 'additional_measurements' in row:
            print(row['id']+' additional: '+json.dumps(row['additional_measurements'],sort_keys=True))
    text += ['', '## Numerical implementation note', '',
             'The corrected Gaussian-distance stabilization is described in `docs/ALGORITHM.md`, under Numerical stability.',
             'Its failing and repaired offset corpus and pre-specified-domain equivalence evidence are in `results/kernel_repair_evidence.json`.',
             'This is an implementation roundoff correction; the faithful arithmetic and the mathematical kernel definition are retained.']
    Path('docs').mkdir(exist_ok=True)
    Path('docs/PORT_DISCREPANCIES.md').write_text('\n'.join(text)+'\n', encoding='utf-8', newline='\n')
    print(f"Diagnostic held-out accuracy (not selected/tuned): {output['probe_accuracy']:.10g}")


if __name__ == '__main__':
    main()
