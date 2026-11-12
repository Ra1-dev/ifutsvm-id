"""Dataset-level Friedman/Nemenyi and exact signed-rank/Holm inference.

Demšar (2006), JMLR 7:1-30, pp. 11-15. Repeated folds are averaged within
dataset, never counted as independent datasets. Larger metric values are better.
"""
from itertools import combinations,product
import numpy as np
from scipy.stats import friedmanchisquare,rankdata,studentized_range

SOURCE='https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf'


def friedman_nemenyi(values,names,alpha=.05):
    values=np.asarray(values,float)
    if values.ndim!=2 or values.shape[1]!=len(names) or len(values)<2 or len(names)<3 or not np.isfinite(values).all():
        raise ValueError('finite dataset-by-model matrix with >=2 datasets and >=3 models required')
    n,k=values.shape
    ranks=np.array([rankdata(-row,method='average') for row in values])
    if np.all(ranks==ranks[:,[0]]): statistic,pvalue=0.,1.
    else:
        test=friedmanchisquare(*values.T); statistic,pvalue=float(test.statistic),float(test.pvalue)
    average=ranks.mean(0); standard_error=np.sqrt(k*(k+1)/(6*n))
    q=float(studentized_range.ppf(1-alpha,k,np.inf)/np.sqrt(2))
    cd=q*standard_error
    pairs=[]
    for i,j in combinations(range(k),2):
        delta=abs(average[i]-average[j])
        p=float(studentized_range.sf(delta/standard_error*np.sqrt(2),k,np.inf))
        pairs.append(dict(first=names[i],second=names[j],rank_difference=float(delta),
                          nemenyi_p=p,exceeds_critical_difference=bool(delta>cd),
                          significant=bool(pvalue<alpha and delta>cd)))
    return dict(datasets=n,models=list(names),alpha=alpha,source=SOURCE,
        per_dataset_values=values.tolist(),per_dataset_ranks=ranks.tolist(),
        friedman_statistic=statistic,friedman_p=pvalue,
        average_ranks=dict(zip(names,average.tolist())),q_alpha=q,critical_difference=float(cd),pairs=pairs,
        posthoc_gate='Nemenyi significance requires rejected omnibus Friedman test; non-significance does not establish equivalence.')


def exact_signed_rank(differences):
    """Two-sided sign enumeration, including ties; zero differences discarded.

    At twelve datasets, all <=4096 sign assignments can be enumerated exactly.
    Rounding at 12 decimal places only removes floating arithmetic artifacts,
    a fixed convention applied before seeing any contrast. Symmetry of paired
    differences is the signed-rank test's distributional assumption.
    """
    raw=np.asarray(differences,float)
    if raw.ndim!=1 or len(raw)>20 or not np.isfinite(raw).all(): raise ValueError('finite <=20 paired differences required')
    d=np.round(raw,12); d=d[d!=0]
    if not len(d): return dict(statistic=0.,pvalue=1.,nonzero=0,assignments=1,zero_method='wilcox')
    ranks=rankdata(np.abs(d),method='average')
    positive=float(ranks[d>0].sum()); total=float(ranks.sum())
    statistic=min(positive,total-positive)
    positive_sums=np.array([np.dot(bits,ranks) for bits in product((0,1),repeat=len(d))])
    null=np.minimum(positive_sums,total-positive_sums)
    p=float(np.mean(null<=statistic+1e-12))
    return dict(statistic=float(statistic),pvalue=p,nonzero=len(d),assignments=len(null),zero_method='wilcox')


def holm(pvalues):
    p=np.asarray(pvalues,float)
    if p.ndim!=1 or not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError('valid p-values required')
    order=np.argsort(p,kind='stable'); adjusted=np.empty(len(p)); running=0.
    for position,index in enumerate(order):
        running=max(running,(len(p)-position)*p[index]); adjusted[index]=min(1.,running)
    return adjusted.tolist()


def paired_family(matrices,names,metrics,alpha=.05):
    """One Holm family across every named metric and model-pair contrast."""
    rows=[]
    for metric in metrics:
        matrix=np.asarray(matrices[metric],float)
        for i,j in combinations(range(len(names)),2):
            differences=matrix[:,i]-matrix[:,j]
            test=exact_signed_rank(differences)
            rows.append(dict(metric=metric,first=names[i],second=names[j],
                mean_difference=float(differences.mean()),median_difference=float(np.median(differences)),
                differences=differences.tolist(),**test))
    for row,adjusted in zip(rows,holm([r['pvalue'] for r in rows])):
        row.update(holm_p=adjusted,significant=adjusted<alpha)
    return dict(alpha=alpha,family='all declared metric/pair contrasts jointly',
        unit='dataset mean over fixed seeds',rounding_decimal_places=12,comparisons=rows,
        assumptions='Independent datasets; symmetric paired-difference distribution for signed-rank interpretation.')
