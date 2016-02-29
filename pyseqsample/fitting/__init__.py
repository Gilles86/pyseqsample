import numpy as np
from copy import copy

class Parameter(object):
    def __init__(self, name, value=0.0):
        self.name = name
        self.value = value

class FreeParameter(Parameter):

    def __repr__(self):
        return "<FreeParameter %s>" % self.name

    def __init(self, *args, **kwargs):
        return super(FreeParameter, self).__init__(*args, **kwargs)


class FixedParameter(Parameter):

    def __init(self, *args, **kwargs):
        return super(FixedParameter, self).__init__(*args, **kwargs)

    def __repr__(self):
        return "<FixedParameter %s>" % self.name

class RaceModelFitter(object):

    def __init__(self, model, parameter_mapping, responses, rts, conditions):


        self.parameter_mapping = parameter_mapping

        self.n_conditions = len(parameter_mapping)
        self.n_accumulators = [len(self.parameter_mapping[c]) for c in xrange(self.n_conditions)]

        self.models = [copy(model) for i in xrange(self.n_conditions)]

        self.free_parameters = []

        for c in xrange(self.n_conditions):
            for a in xrange(self.n_accumulators[c]):
                for key, node in parameter_mapping[c][a].items():
                    if type(node) == FixedParameter:
                        self.models[c].accumulators[a].params[key] = node.value
                    elif node not in self.free_parameters:
                        self.free_parameters.append(node)

        self.free_parameters = sorted(self.free_parameters, key= lambda x: x.name)
        self.n_parameters = len(self.free_parameters)

        self.conditions = conditions
        assert(self.n_conditions == len(np.unique(self.conditions)))


        self.responses = responses
        self.rts = rts



    def likelihood(self, parameter_vector):

        total_likelihood = 0

        for i, p in enumerate(parameter_vector):
            self.free_parameters[i].value = p

        for c in xrange(self.n_conditions):
            for acc in xrange(self.n_accumulators[c]):
                for parameter, node in self.parameter_mapping[c][acc].items():
                    self.models[c].accumulators[acc].params[parameter] = node.value

            total_likelihood += np.sum(self.models[c].likelihood_(self.responses, self.rts))

        return total_likelihood



