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
    
    def pdf(self, t):
        return pdf(t=t, **self.params)
    
    def cdf(self, t):
        return cdf(t=t, **self.params)  
    
    def sample_finishing_times(self, n=1000):
        starting_points = sp.stats.uniform(0, self.params['A']).rvs(n)
        speeds = sp.stats.norm(self.params['v'], self.params['sv']).rvs(n)
        
        finishing_times = (self.params['b'] - starting_points) / speeds + self.params['ter']
        
        
        return finishing_times
        
        
class LBAAccumulatorProbabilistic(Accumulator):
    
    accumulator_parameters = ['ter', 'A', 'v', 'sv', 'b', 'p']
    
    def __init__(self, ter=.2, A=.5, v=1, sv=.1, b=2, p=0.5, B=None):
        
        if B != None:
            b = A + B
        
        super(LBAAccumulatorProbabilistic, self).__init__(ter=ter, A=A, v=v, sv=sv, b=b, p=p)
    
    def pdf(self, t):
        return self.params['p'] * pdf(t=t, 
                                                             ter=self.params['ter'],
                                                             A=self.params['A'],
                                                             v=self.params['v'],
                                                             sv=self.params['sv'],
                                                             b=self.params['b']
                                                             )
    
    def cdf(self, t):
        return self.params['p'] * cdf(t=t, 
                                                             ter=self.params['ter'],
                                                             A=self.params['A'],
                                                             v=self.params['v'],
                                                             sv=self.params['sv'],
                                                             b=self.params['b']
                                                             )
    
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

