from pyseqsample.utils import dnormP, pnormP
import numpy as np
from pyseqsample.models.racemodels import Accumulator
import scipy as sp
from scipy import stats


class LBAAccumulator(Accumulator):
    
    accumulator_parameters = ['ter', 'A', 'v', 'sv', 'b']
    
    def __init__(self, ter=.2, A=.5, v=1, sv=.1, b=2, B=None):
        
        if B != None:
            b = A + B
        
        super(LBAAccumulator, self).__init__(ter=ter, A=A, v=v, sv=sv, b=b)
    
    def pdf(self, t, condition=0):
       
        if isinstance(condition, int):
            conditions = np.tile(condition, len(t))

        density = np.zeros_like(t)

        for cond in np.arange(self.n_conditions):
            idx = conditions == cond
            density[idx] =  pdf(t=t[idx], *self.params[cond, :].tolist())

        return density
    
    def cdf(self, t, condition=0):
        if isinstance(condition, int):
            conditions = np.tile(condition, len(t))

        density = np.zeros_like(t)

        for cond in np.arange(self.n_conditions):
            idx = conditions == cond
            density[idx] =  cdf(t=t[idx], *self.params[cond, :].tolist())

        return density
    
    def sample_finishing_times(self, condition=None, n=1000, robust=True):

        if condition is None:
            condition = np.repeat(np.arange(self.n_conditions), n)

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

        print speeds

        finishing_times = (self.get_param('b', condition=condition) - starting_points) / speeds + self.get_param('ter', condition=condition)
        
        
        return finishing_times
        
        
class LBAAccumulatorProbabilistic(Accumulator):
    
    accumulator_parameters = ['ter', 'A', 'v', 'sv', 'b', 'p']
    
    def __init__(self, ter=.2, A=.5, v=1, sv=.1, b=2, p=0.5, B=None):
        
        if B != None:
            b = A + B
        
        super(LBAAccumulatorProbabilistic, self).__init__(ter=ter, A=A, v=v, sv=sv, b=b, p=p)
    

    def pdf(self, t, condition=0):
       
        if isinstance(condition, int):
            conditions = np.tile(condition, len(t))

        density = np.zeros_like(t)

        for cond in np.arange(self.n_conditions):
            idx = conditions == cond
            density[idx] = self.params[cond, -1] * pdf(t=t[idx], *self.params[cond, :-1].tolist())

        return density
    
    def cdf(self, t, condition=0):
        if isinstance(condition, int):
            conditions = np.tile(condition, len(t))

        density = np.zeros_like(t)

        for cond in np.arange(self.n_conditions):
            idx = conditions == cond
            density[idx] = self.params[cond, -1] * cdf(t=t[idx], *self.params[cond, :-1].tolist())

        return density

    
    def sample_finishing_times(self, n=1000):
        starting_points = sp.stats.uniform(0, self.params['A']).rvs(n)
        speeds = sp.stats.norm(self.params['v'], self.params['sv']).rvs(n)
        
        finishing_times = (self.params['b'] - starting_points) / speeds + self.params['ter']
        
        finishing_times[sp.stats.uniform(0, 1).rvs(n) > self.params['p']] = np.nan
        
        
        return finishing_times
        
        

def pdf(ter, A, v, sv, b, t):
    """LBA PDF for a single accumulator"""
    t=np.maximum(t-ter, 1e-5) # absorbed into pdf 
    if A<1e-10: # LATER solution
        return np.maximum(1e-10, (b/(t**2)*dnormP(b/t, mean=v,sd=sv))
                          /pnormP(v/sv) )
    zs=t*sv
    zu=t*v
    bminuszu=b-zu
    bzu=bminuszu/zs
    bzumax=(bminuszu-A)/zs
    return np.maximum(1e-10, ((v*(pnormP(bzu)-pnormP(bzumax)) +
                    sv*(dnormP(bzumax)-dnormP(bzu)))/A)/pnormP(v/sv))

def cdf(ter, A, v, sv, b, t):
    """LBA CDF for a single accumulator"""
    t=np.maximum(t-ter, 1e-5) # absorbed into cdf         
    if A<1e-10: # LATER solution
        return np.minimum(1 - 1e-10, np.maximum(1e-10, (pnormP(b/t,mean=v,sd=sv))
                                        /pnormP(v/sv) ))
    zs=t*sv
    zu=t*v
    bminuszu=b-zu
    xx=bminuszu-A
    bzu=bminuszu/zs
    bzumax=xx/zs
    tmp1=zs*(dnormP(bzumax)-dnormP(bzu))
    tmp2=xx*pnormP(bzumax)-bminuszu*pnormP(bzu)
    return np.minimum(np.maximum(1e-10,(1+(tmp1+tmp2)/A)/pnormP(v/sv)), 1 - 1e-10)

