from pyseqsample.utils import dnormP, pnormP, dnorm, pnorm
import numpy as np
from pyseqsample.models.racemodels import Accumulator
import scipy as sp
from scipy import stats


class LBAAccumulator(Accumulator):
    
    accumulator_parameters = ['ter', 'A', 'v', 'sv', 'b']
    
    def __init__(self, ter=.2, A=.5, v=1, sv=.1, b=2, B=None, *args, **kwargs):
        
        if B != None:
            b = A + B
        
        return super(LBAAccumulator, self).__init__(ter=ter, A=A, v=v, sv=sv, b=b, *args, **kwargs)
    
    def pdf(self, t, condition=0):

        if self.params.shape[0] == 1:
            return pdf(t=t, *self.params[0, :].tolist())
       
        if isinstance(condition, int):
            condition = np.tile(condition, len(t))

        density = np.zeros_like(t)

        for cond in np.arange(self.n_conditions):
            idx = condition == cond
            density[idx] =  pdf(t=t[idx], *self.params[cond, :].tolist())

        return density
    
    def cdf(self, t, condition=0):

        if self.params.shape[0] == 1:
            return cdf(t=t, *self.params[0, :].tolist())

        if isinstance(condition, int):
            condition = np.tile(condition, len(t))

        density = np.zeros_like(t)

        for cond in np.arange(self.n_conditions):
            idx = condition == cond
            density[idx] =  cdf(t=t[idx], *self.params[cond, :].tolist())

        return density
    
    def sample_finishing_times(self, condition=None, n=1000, robust=True):
        
        if condition is None:
            condition = np.repeat(np.arange(self.n_conditions), n)

        if isinstance(condition, int):
            condition = np.repeat(condition, n)

        starting_points = sp.stats.uniform(0, self.get_param('A', condition=condition)).rvs(len(condition))

        if robust:
            speeds = np.zeros(condition.shape)
            for cond in np.unique(condition):
                idx = cond == condition
                loc = self.get_param('v', condition=cond)
                scale = self.get_param('sv', condition=cond)
                speeds[idx] = sp.stats.truncnorm(a=-loc/scale,
                                                 b=np.inf,
                                                 loc=loc, 
                                                 scale=scale).rvs(idx.sum())
        else:
            speeds = sp.stats.norm(self.get_param('v', condition=condition), self.get_param('sv', condition=condition)).rvs(len(condition))


        finishing_times = (self.get_param('b', condition=condition) - starting_points) / speeds + self.get_param('ter', condition=condition)
        
        
        return condition, finishing_times
        
        
class LBAAccumulatorProbabilistic(LBAAccumulator):
    
    accumulator_parameters = ['ter', 'A', 'v', 'sv', 'b', 'p']
    
    def __init__(self, ter=.2, A=.5, v=1, sv=.1, b=2, p=0.5, B=None, **kwargs):
        
        if B != None:
            b = A + B
        
        super(LBAAccumulatorProbabilistic, self).__init__(ter=ter, A=A, v=v, sv=sv, b=b, p=p, **kwargs)
    

    def pdf(self, t, condition=0, robust=False):
       
        if self.params.shape[0] == 1:
            return pself.params[cond, -1] * pdf(t=t, *self.params[0, :].tolist(), robust=robust)

        if isinstance(condition, int):
            condition = np.tile(condition, len(t))


        density = np.zeros_like(t)

        for cond in np.arange(self.n_conditions):
            idx = condition == cond
            density[idx] = self.params[cond, -1] * pdf(t=t[idx], *self.params[cond, :-1].tolist(), robust=robust)

        return density
    
    def cdf(self, t, condition=0, robust=False):

        if self.params.shape[0] == 1:
            return pself.params[cond, -1] * cdf(t=t, *self.params[0, :].tolist(), robust=robust)

        if isinstance(condition, int):
            condition = np.tile(condition, len(t))

        density = np.zeros_like(t)

        for cond in np.arange(self.n_conditions):
            idx = condition == cond
            density[idx] = self.params[cond, -1] * cdf(t=t[idx], *self.params[cond, :-1].tolist(), robust=robust)

        return density

    
    def sample_finishing_times(self, *args, **kwargs):

        condition, finishing_times = super(LBAAccumulatorProbabilistic, self).sample_finishing_times(*args, **kwargs)
        rand_uni = sp.stats.uniform(0, 1).rvs(finishing_times.shape[0])

        for cond in np.unique(condition):
            idx = condition == cond
            finishing_times[idx & (rand_uni > self.get_param('p', condition=cond))] = np.inf
        
        
        return condition, finishing_times
        
        

def pdf(ter, A, v, sv, b, t, robust=False):
    """LBA PDF for a single accumulator"""
    
    if robust:
        dnorm1 = dnormP
        pnorm1 = pnormP
    else:
        dnorm1 = dnorm
        pnorm1 = pnorm


    t=np.maximum(t-ter, 1e-5) # absorbed into pdf 
    if A<1e-10: # LATER solution
        return np.maximum(1e-10, (b/(t**2)*dnorm1(b/t, mean=v,sd=sv))
                          /pnorm1(v/sv) )
    zs=t*sv
    zu=t*v
    bminuszu=b-zu
    bzu=bminuszu/zs
    bzumax=(bminuszu-A)/zs
    return np.maximum(1e-10, ((v*(pnorm1(bzu)-pnorm1(bzumax)) +
                    sv*(dnorm1(bzumax)-dnorm1(bzu)))/A)/pnorm1(v/sv))

def cdf(ter, A, v, sv, b, t, robust=False):

    """LBA CDF for a single accumulator"""

    if robust:
        dnorm1 = dnormP
        pnorm1 = pnormP
    else:
        dnorm1 = dnorm
        pnorm1 = pnorm

    t=np.maximum(t-ter, 1e-5) # absorbed into cdf         
    if A<1e-10: # LATER solution
        return np.minimum(1 - 1e-10, np.maximum(1e-10, (pnorm1(b/t,mean=v,sd=sv))
                                        /pnorm1(v/sv) ))
    zs=t*sv
    zu=t*v
    bminuszu=b-zu
    xx=bminuszu-A
    bzu=bminuszu/zs
    bzumax=xx/zs
    tmp1=zs*(dnorm1(bzumax)-dnorm1(bzu))
    tmp2=xx*pnorm1(bzumax)-bminuszu*pnorm1(bzu)
    return np.minimum(np.maximum(1e-10,(1+(tmp1+tmp2)/A)/pnorm1(v/sv)), 1 - 1e-10)

