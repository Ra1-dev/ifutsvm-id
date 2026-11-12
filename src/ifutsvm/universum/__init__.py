"""Minority Gaussian conditioned on equal fitted class-conditional density.

Public strategy, passed through IFUTSVMID(universum=callable). All statistics
come from the two supplied training-class arrays. No class-prior estimate enters
the conditioning equation. See docs/THEORY.md for scope and caveats.
"""
from dataclasses import dataclass
import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class ConditionalGaussianUniversum:
    """Training-only conditional Gaussian sampler.

    Call with (minority_rows, majority_rows, count, numpy.random.Generator).
    The density-neutral hyperplane is fitted from the supplied rows only.
    ``parameters`` exposes the fitted law for support and moment audits.
    The fixed relative ridge regularizes both class covariance estimates.
    """
    covariance_ridge: float = 1e-8

    def parameters(self,minority,majority):
        minority=np.asarray(minority,float); majority=np.asarray(majority,float)
        if (minority.ndim!=2 or majority.ndim!=2 or minority.shape[1]!=majority.shape[1]
            or not minority.shape[1] or not len(minority) or not len(majority) or not np.isfinite(minority).all()
            or not np.isfinite(majority).all()):
            raise ValueError('finite nonempty training classes with matching features required')
        if not np.isfinite(self.covariance_ridge) or self.covariance_ridge<=0:
            raise ValueError('covariance_ridge must be positive')
        p=minority.shape[1]
        def covariance(X):
            return LedoitWolf().fit(X).covariance_ if len(X)>1 else np.zeros((p,p))
        # Equal class weighting prevents the abundant class from determining
        # the covariance solely through its sample count.
        sigma=(covariance(minority)+covariance(majority))/2
        scale=max(float(np.trace(sigma)/p),np.finfo(float).eps)
        sigma+=self.covariance_ridge*scale*np.eye(p)
        mean1,mean2=minority.mean(0),majority.mean(0)
        midpoint=(mean1+mean2)/2; delta=mean2-mean1
        normal=np.linalg.solve(sigma,delta)
        separation=float(delta@normal)
        direction=delta; conditioning_normal=normal; conditioning_separation=separation
        magnitude=float(np.max(np.abs(delta)))
        if magnitude>0 and (separation==0 or magnitude<np.sqrt(np.finfo(float).tiny)):
            # The law depends on direction, not separation magnitude. Scale
            # before the outer product loses precision in the subnormal range,
            # including when delta.T @ solve(Sigma, delta) is still positive.
            # Ordinary representable training environments retain their arithmetic.
            direction=delta/np.max(np.abs(delta))
            conditioning_normal=np.linalg.solve(sigma,direction)
            conditioning_separation=float(direction@conditioning_normal)
        if conditioning_separation>0:
            conditional=sigma-np.outer(direction,direction)/conditioning_separation
        else:
            conditional=sigma.copy()
        conditional=(conditional+conditional.T)/2
        return dict(mean1=mean1,mean2=mean2,midpoint=midpoint,sigma=sigma,
                    normal=normal,separation=separation,conditional_covariance=conditional,
                    conditioning_normal=conditioning_normal,conditioning_direction=direction,
                    conditioning_separation=conditioning_separation)

    def __call__(self,minority,majority,count,rng):
        if not isinstance(count,(int,np.integer)) or count<0:
            raise ValueError('count must be a nonnegative integer')
        t=self.parameters(minority,majority)
        values,vectors=np.linalg.eigh(t['conditional_covariance'])
        root=vectors*np.sqrt(np.maximum(values,0))[None,:]
        U=t['midpoint']+rng.normal(size=(count,len(values)))@root.T
        if t['conditioning_separation']>0:
            # Remove numerical leakage into the conditioned direction. This is
            # the covariance-metric projection, not a data-dependent rejection.
            residual=(U-t['midpoint'])@t['conditioning_normal']
            U-=np.outer(residual/t['conditioning_separation'],t['conditioning_direction'])
        return U

__all__ = ["ConditionalGaussianUniversum"]
