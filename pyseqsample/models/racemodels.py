import numpy as np
from collections import OrderedDict

class Accumulator(object):
    
    accumulator_parameters = []
    
    def __init__(self, *args, **kwargs):
        
        self.params = OrderedDict()

        
        for param in self.accumulator_parameters:
            self.params[param] = kwargs[param]

    def set_params(self, params):
        self.params = OrderedDict(zip(self.params.keys(), params))

    
    def pdf(self):
        pass
    
    def cdf(self):
        pass    
    
    def sample_finishing_time(self):
        pass

class RaceModel(object):
    
    def __init__(self, accumulators, accumulator2response=None):        
        self.accumulators = accumulators
        self.n_accumulators = len(accumulators)

        if accumulator2response == None:
            self.accumulator2response = np.arange(self.n_accumulators) + 1
        else:
            self.accumulator2response = np.array(accumulator2response)

        
    
    def likelihood_(self, responses, rts, log=True):
        
       
        assert(responses.shape == rts.shape)

        if log:
            likelihood = np.zeros_like(rts)
        else:
            likelihood = np.ones_like(rts)
        
        for i in np.arange(self.n_accumulators):
            #print self.accumulator2response
            idx = responses == self.accumulator2response[i]

            #rts = rts[idx]
            
            if log:
                likelihood[idx] = np.log(self.accumulators[i].pdf(rts[idx]))
            else:
                likelihood[idx] = self.accumulators[i].pdf(rts[idx])
            
            for j in np.arange(self.n_accumulators):
                if i != j:
                    if log:
                        likelihood[idx] += np.log(1 - self.accumulators[j].cdf(rts[idx])) 
                    else:
                        likelihood[idx] *= (1 -  self.accumulators[j].cdf(rts[idx]))
                    
        return likelihood
    
    
    def sample_responses(self, n=1000, remove_negative_finishing_times=True):
        
        finishing_times = np.zeros((n, self.n_accumulators))
        
        for i, accumulator in enumerate(self.accumulators):
            finishing_times[:, i] = accumulator.sample_finishing_times(n=n)

        if remove_negative_finishing_times:
            finishing_times[finishing_times < 0] = np.nan
            
        rts = np.nanmin(finishing_times, 1)
        responses = self.accumulator2response[np.nanargmin(finishing_times, 1)]


        #responses = responses[rts > 0]
        #rts = rts[rts > 0]
        
        return responses, rts


        
        
class RaceModelProbabilisticAccumulators(RaceModel):

    def __init__(self, *args, **kwargs):

        super(RaceModelProbabilisticAccumulators, self).__init__(*args, **kwargs)

        self.p_accumulators = [acc for acc in self.accumulators if 'p' in acc.params]




            
            
        
        
        
        
