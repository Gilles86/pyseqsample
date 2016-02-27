
class Accumulator(object):
    
    accumulator_parameters = []
    
    def __init__(self, *args, **kwargs):
        
        self.params = {}
        
        for param in self.accumulator_parameters:
            self.params[param] = kwargs[param]
    
    def pdf(self):
        pass
    
    def cdf(self):
        pass    
    
    def sample_finishing_time(self):
        pass

class RaceModel:
    
    def __init__(self, accumulators):        
        self.accumulators = accumulators
        self.n_accumulators = len(accumulators)
        
    
    def likelihood_(self, responses, rts, log=True):
        
        
        if log:
            likelihood = np.zeros_like(rts)
        else:
            likelihood = np.ones_like(rts)
        
        for i in np.arange(self.n_accumulators):
            idx = responses == (i + 1)
            rts = rts[idx]
            
            if log:
                likelihood[idx] = np.log(self.accumulators[i].pdf(rts[idx]))
            else:
                likelihood[idx] = self.accumulators[i].pdf(rts[idx])
            
            for j in np.arange(self.n_accumulators):
                if i != j:
                    if log:
                        likelihood[idx] += np.log(self.accumulators[i].cdf(rts[idx])) 
                    else:
                        likelihood[idx] *= self.accumulators[i].cdf(rts[idx]) 
                    
        return likelihood
    
    
    def sample_responses(self, n=1000):
        
        finishing_times = np.zeros((n, self.n_accumulators))
        
        for i, accumulator in enumerate(self.accumulators):
            finishing_times[:, i] = accumulator.sample_finishing_times(n=n)
            
        rts = np.nanmin(finishing_times, 1)
        responses = np.nanargmin(finishing_times, 1) + 1
        
        return responses, rts
        
        
               
            
            
        
        
        
        
