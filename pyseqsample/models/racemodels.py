import numpy as np
from collections import OrderedDict
from collections import Iterable
import scipy as sp

class Accumulator(object):
    
    accumulator_parameters = []
    
    def __init__(self, *args, **kwargs):


        # Check if there are multiple conditions involved:
        self.n_conditions = 1
        
        for param in self.accumulator_parameters:
            if isinstance(kwargs[param], Iterable):
                self.n_conditions = np.max((self.n_conditions, len(kwargs[param])))
        
        self.params = np.zeros((self.n_conditions, len(self.accumulator_parameters)))
        
        for i, param in enumerate(self.accumulator_parameters):
            self.params[:, i] = kwargs[param]

        if 'name' in kwargs.keys():
            self.name = kwargs['name']


    def set_params(self, params):
        self.params = OrderedDict(zip(self.params.keys(), params))

    
    def pdf(self):
        pass
    
    def cdf(self):
        pass    
    
    def sample_finishing_times(self):
        pass

    def get_param(self, param, condition=slice(None)):
        if self.n_conditions == 1:
            return self.params[0, self.accumulator_parameters.index(param)]
        else:
            return self.params[condition, self.accumulator_parameters.index(param)]

class RaceModel(object):
    
    def __init__(self, accumulators, accumulator2response=None):        
        self.accumulators = accumulators
        self.n_accumulators = len(accumulators)

        if accumulator2response is None:
            self.accumulator2response = np.arange(self.n_accumulators) + 1
            self.more_accumulators_than_responses = False
        else:
            self.accumulator2response = np.array(accumulator2response)
            self.more_accumulators_than_responses = True


        self.n_conditions = np.max([acc.n_conditions for acc in self.accumulators])

        
    
    def likelihood_(self, responses, rts, condition=None, log=True, more_acc_than_resp=True):
        
       
        assert(responses.shape == rts.shape)

        if condition is None:
            print "WARNING: no conditions given"
            np.tile(0, len(responses))


        if self.more_accumulators_than_responses:
            likelihood = np.zeros((rts.shape[0], self.n_accumulators))
        else:
            if log:
                likelihood = np.zeros_like(rts)
            else:
                likelihood = np.ones_like(rts)
        
        if self.more_accumulators_than_responses:

            #MORE THAN  ONE ACCUMULATOR PER RESPONE

            test = np.zeros_like(rts)
            for i in np.arange(self.n_accumulators):
                idx = responses == self.accumulator2response[i]


                likelihood[idx, i] = self.accumulators[i].pdf(rts[idx], condition=condition[idx])

                
                for j in np.arange(self.n_accumulators):
                    if i != j:
                        likelihood[idx, i] *= (1 -  self.accumulators[j].cdf(rts[idx], condition=condition[idx]))
            
            likelihood = np.sum(likelihood, 1)


            if log:
                likelihood = np.log(likelihood)

        else:

            # ONLY ONE ACCUMULATOR PER RESPONSE
            for i in np.arange(self.n_accumulators):
                idx = responses == self.accumulator2response[i]

                
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
    
    
    def sample_responses(self, n=1000, condition=None, remove_negative_finishing_times=True):
        
        if condition is None:
            condition = np.arange(self.n_conditions)

        condition = np.repeat(condition, n)

        n = condition.shape[0]

        finishing_times = np.zeros((n, self.n_accumulators))

        for i, accumulator in enumerate(self.accumulators):
            _, finishing_times[:, i] = accumulator.sample_finishing_times(condition=condition)

        if remove_negative_finishing_times:
            finishing_times[finishing_times < 0] = np.nan
            
        rts = np.nanmin(finishing_times, 1)

        responses = np.nanargmin(finishing_times, 1)
        
        mapped_responses = np.ones_like(responses) * np.nan
        mapped_responses[~np.isnan(responses)] = self.accumulator2response[responses[~np.isnan(responses)]]


        return condition, mapped_responses, rts


    def get_quantiles(self, condition, response, q=(0.1, 0.3, 0.5, 0.7, 0.9), tmax=10, steps=5000):

        t = np.linspace(0, tmax, steps)

        y = self.likelihood_(np.ones_like(t) * response, t, condition=np.ones_like(t) * condition, log=False)
        integral = sp.integrate.cumtrapz(y, t, initial=0)

        q = np.array(q)

        rts = t[np.abs(integral - q[:, np.newaxis]).argmin(1)]

        return rts

    def get_response_proportion(self, condition, response, tmax=np.inf):

        ll = lambda t: self.likelihood_(np.ones((1,)) * response, np.array([t]), condition=np.ones((1,)) * condition, log=False)[0]
        return sp.integrate.quad(ll, 0, tmax)[0]



        
class RaceModelProbabilisticAccumulators(RaceModel):

    def __init__(self, *args, **kwargs):

        super(RaceModelProbabilisticAccumulators, self).__init__(*args, **kwargs)

        self.p_accumulators = [acc for acc in self.accumulators if 'p' in acc.params]




            
            
        
        
        
        
