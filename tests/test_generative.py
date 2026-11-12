"""seeded sampling, neutral support, degenerate shapes and fit integration."""
import numpy as np
import pytest
from ifutsvm import IFUTSVMID
from ifutsvm.universum import ConditionalGaussianUniversum


@pytest.mark.parametrize('features',[1,3,11])
def test_conditioned_sampling_contract(features):
    rng=np.random.default_rng(591)
    A=rng.normal(size=(7,features))-0.7; B=rng.normal(size=(33,features))+.4
    before=A.copy(); generator=ConditionalGaussianUniversum()
    U=generator(A,B,53,np.random.default_rng(71))
    np.testing.assert_array_equal(U,generator(A,B,53,np.random.default_rng(71)))
    np.testing.assert_array_equal(A,before)
    assert U.shape==(53,features) and np.isfinite(U).all()
    t=generator.parameters(A,B)
    assert np.max(np.abs((U-t['midpoint'])@t['normal']))<1e-10
    if features==1: np.testing.assert_allclose(U,np.tile(t['midpoint'],(53,1)),atol=1e-12)


@pytest.mark.parametrize('same_means',[True,False])
def test_constant_classes_and_singletons(same_means):
    A=np.zeros((1,4)); B=np.ones((5,4))*(not same_means)
    U=ConditionalGaussianUniversum()(A,B,17,np.random.default_rng(1))
    assert U.shape==(17,4) and np.isfinite(U).all()


def test_generator_inside_paper_fit():
    from sklearn.base import clone
    rng=np.random.default_rng(19); X=rng.normal(size=(34,3)); y=np.r_[np.ones(8),np.zeros(26)]
    model=IFUTSVMID(universum=ConditionalGaussianUniversum(),random_state=12).fit(X,y)
    tr=model.trace_; t=model.universum.parameters(tr['X1'],tr['X2'])
    assert np.max(np.abs((tr['U']-t['midpoint'])@t['normal']))<1e-10
    assert set(model.predict(X))<=set(y)
    assert type(model.universum).__module__=='ifutsvm.universum'
    replay=clone(model).fit(X,y)
    for key in model.trace_:np.testing.assert_array_equal(model.trace_[key],replay.trace_[key])


@pytest.mark.parametrize('shift',[1e-5,1e-9,1e-12])
def test_distinct_nearby_means_still_condition_to_point_in_one_dimension(shift):
    A=np.array([[-1.],[1.]]); B=A+shift
    generator=ConditionalGaussianUniversum(); t=generator.parameters(A,B)
    assert t['separation']>0
    U=generator(A,B,100,np.random.default_rng(47))
    np.testing.assert_allclose(U,np.tile(t['midpoint'],(100,1)),rtol=0,atol=1e-14)


def test_underflowed_separation_preserves_conditioning_direction():
    A=np.zeros((3,2)); B=np.tile([1e-200,0.],(7,1))
    generator=ConditionalGaussianUniversum(); t=generator.parameters(A,B)
    U=generator(A,B,100,np.random.default_rng(47))
    assert t['separation']==0 and t['conditioning_separation']>0
    np.testing.assert_allclose(U[:,0],t['midpoint'][0],rtol=0,atol=1e-25)
    assert np.std(U[:,1])>0


@pytest.mark.parametrize('exponent',[155,160,165,170,175,200])
def test_subnormal_outer_product_preserves_conditional_covariance(exponent):
    A=np.zeros((3,2));B=np.tile([10.**(-exponent),0.],(7,1))
    generator=ConditionalGaussianUniversum();t=generator.parameters(A,B)
    # Independent limiting law: x is fixed at the midpoint, y retains Sigma_yy.
    expected=np.diag([0.,t['sigma'][1,1]])
    np.testing.assert_allclose(t['conditional_covariance']/t['sigma'][1,1],
        expected/t['sigma'][1,1],rtol=0,atol=1e-12)
    U=generator(A,B,100,np.random.default_rng(47))
    np.testing.assert_allclose(U[:,0],t['midpoint'][0],rtol=0,
        atol=np.sqrt(t['sigma'][0,0])*1e-12)
    assert np.std(U[:,1])>0


def test_large_sample_moments_against_independent_conditional_law():
    """Gaussian mean and Wishart covariance sampling-error bounds."""
    from scipy.stats import chi2
    from sklearn.covariance import LedoitWolf
    rng=np.random.default_rng(583)
    transform=np.array([[1.,.3,0],[0,1.8,.2],[.1,0,.6]])
    A=rng.normal(size=(43,3))@transform-np.array([.5,.1,.2])
    B=rng.normal(size=(129,3))@transform+np.array([.5,.1,.2])
    sigma=(LedoitWolf().fit(A).covariance_+LedoitWolf().fit(B).covariance_)/2
    sigma+=1e-8*max(np.trace(sigma)/3,np.finfo(float).eps)*np.eye(3)
    delta=B.mean(0)-A.mean(0); a=np.linalg.solve(sigma,delta)
    midpoint=(A.mean(0)+B.mean(0))/2
    target=sigma-np.outer(delta,delta)/(delta@a)
    n=50000
    U=ConditionalGaussianUniversum()(A,B,n,np.random.default_rng(584))
    np.testing.assert_allclose((U-midpoint)@a,0,atol=1e-10)
    values,vectors=np.linalg.eigh(target); keep=values>1e-10
    mean=U.mean(0)-midpoint
    standardized_mean=n*np.sum((vectors[:,keep].T@mean)**2/values[keep])
    assert standardized_mean<chi2.ppf(1-1e-6,np.sum(keep))
    # E||sample_cov-Sigma||_F^2=(tr(Sigma)^2+||Sigma||_F^2)/(n-1).
    covariance_rms=np.sqrt((np.trace(target)**2+np.linalg.norm(target)**2)/(n-1))
    assert np.linalg.norm(np.cov(U,rowvar=False)-target)<8*covariance_rms


def test_research_default_is_identical_and_code_construction_matches_reference():
    from experiments.research_model import ResearchIFUTSVMID,PublishedUniversum
    from ifutsvm.reference import trace_reference
    rng=np.random.default_rng(44)
    X=rng.normal(size=(28,3)); y=np.r_[np.ones(8),-np.ones(20)]
    paper=IFUTSVMID(random_state=71).fit(X,y)
    research=ResearchIFUTSVMID(random_state=71).fit(X,y)
    for key in paper.trace_: np.testing.assert_array_equal(paper.trace_[key],research.trace_[key])
    reference=trace_reference(np.column_stack((X,y)),np.column_stack((X,y)),.7,1.,.3)
    generated=PublishedUniversum()(X[y==1],X[y==-1],12,np.random.default_rng(71))
    np.testing.assert_array_equal(generated,reference['U'])


@pytest.mark.parametrize('size,fuzzy,side',[(.25,False,'symmetric'),(2.,True,'paper_dual'),(1.,False,'paper_dual')])
def test_research_variants_against_direct_primal(size,fuzzy,side):
    import cvxpy as cp
    from experiments.research_model import ResearchIFUTSVMID
    from ifutsvm._numeric import aug,rbf
    rng=np.random.default_rng(45); X=rng.normal(size=(28,3)); y=np.r_[np.ones(8),-np.ones(20)]
    model=ResearchIFUTSVMID(C=.7,C3=.4,Cu=.3,mu=1.2,eps=.3,random_state=16,
        universum=ConditionalGaussianUniversum(),size_multiplier=size,if_scores=fuzzy,universum_side2=side).fit(X,y)
    t=model.trace_
    assert len(t['U'])==int(np.ceil(12*size))
    assert len(t['U_star'])==min(4,len(t['U']))
    if not fuzzy:
        np.testing.assert_array_equal(t['s_min'],np.ones(8))
        np.testing.assert_array_equal(t['s_maj'],np.ones(20))
    designs=[aug(rbf(t[k],t['D'],1.2)) for k in ('X1','X2','X2_star','U','U_star')]
    E,F,Fs,G,Gs=designs
    for key,P,Q,U,weights,sign,rhs in [('z1',E,-Fs,Gs,t['s_maj_star'],1,-.7),
            ('z2',F,E,G,t['s_min'],-1 if side=='symmetric' else 1,-.7 if side=='symmetric' else .7)]:
        z=cp.Variable(P.shape[1]); xi=cp.Variable(len(Q),nonneg=True); lam=cp.Variable(len(U),nonneg=True)
        objective=.5*cp.sum_squares(P@z)+.2*cp.sum_squares(z)+.7*weights@xi+.3*cp.sum(lam)
        problem=cp.Problem(cp.Minimize(objective),[Q@z>=1-xi,sign*U@z>=rhs-lam])
        problem.solve(solver='CLARABEL',tol_gap_abs=1e-12,tol_gap_rel=1e-12,tol_feas=1e-12,max_iter=1000)
        assert np.linalg.norm(t[key]-z.value)/np.linalg.norm(z.value)<1e-5


def test_statistics_ties_null_and_independent_oracle():
    from scipy.stats import wilcoxon,PermutationMethod
    from experiments.statistics import exact_signed_rank,holm,friedman_nemenyi
    for d in (np.array([1.,-2,3,4,-5,6]),np.array([0.,1,1,-1,2,-2])):
        expected=wilcoxon(d,method=PermutationMethod(n_resamples=np.inf),zero_method='wilcox')
        measured=exact_signed_rank(d)
        np.testing.assert_allclose(measured['statistic'],expected.statistic)
        np.testing.assert_allclose(measured['pvalue'],expected.pvalue)
    assert exact_signed_rank(np.zeros(12))['pvalue']==1.
    np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
    result=friedman_nemenyi(np.ones((12,7)),[str(i) for i in range(7)])
    assert result['friedman_p']==1.
    # Independently integrate the range of seven standard normal draws.
    # The historical printed table is an approximation, not a numerical oracle.
    from scipy.integrate import quad
    from scipy.stats import norm
    width=result['q_alpha']*np.sqrt(2)
    cdf,error=quad(lambda x:7*norm.pdf(x)*(norm.cdf(x+width)-norm.cdf(x))**6,
                   -np.inf,np.inf,epsabs=1e-11,epsrel=1e-11)
    assert error<1e-9
    np.testing.assert_allclose(cdf,.95,rtol=0,atol=1e-9)
    nontrivial=friedman_nemenyi([[3,2,1],[3,2,1],[1,3,2]],['A','B','C'])
    np.testing.assert_allclose(nontrivial['friedman_statistic'],2.)
    np.testing.assert_allclose(nontrivial['friedman_p'],np.exp(-1.))
