from dopamine.fapprox.linear import Linear 
from dopamine.tools.dataset import Dataset
import numpy as np
import operator

class ResampledLinear(Linear):


    def reset(self):
        """ this initializes the function approximator to an initial state,
            forgetting everything it has learned before. """
        Linear.reset(self)
        self.dataset.clear()
        self.weights = np.array([])


    def update(self, inp, tgt):
        """ update the function approximator to return something closer
            to target when queried for input next time. Some function
            approximators only collect the input/target tuples here and
            learn only on a call to learn(). """
        self.dataset.append(inp, tgt)
        self.weights = np.r_[self.weights, np.array([1])]


    def train(self):
        """ some function approximators learn offline after collecting all 
            samples. this function executes one such learning step. """
        if len(self.dataset) == 0:
            return
            
        inputs = np.c_[self.dataset.inputs, np.ones((len(self.dataset), 1))]
        weights = np.diag(self.weights)
        
        self.matrix = np.dot(np.dot(np.dot(np.linalg.pinv(np.dot(np.dot(inputs.T, weights), inputs)), inputs.T), weights), self.dataset.targets)  
        self.prune()
                

    def prune(self):
        
        # nsamples has to be at least self.indim+1
        nsamples = self.indim*2
        
        input_min = np.min(self.dataset.inputs, axis=0)
        input_max = np.max(self.dataset.inputs, axis=0)
        
        inputs = np.random.uniform(input_min, input_max, size=(nsamples, self.indim))
        # inputs = np.eye(self.indim)
        # inputs = np.r_[np.zeros((1,inputs.shape[1])), inputs]
        outputs = np.dot(np.c_[inputs, np.ones(inputs.shape[0])], self.matrix)
        
        self.weights = np.ones(nsamples) # * (sum(self.weights) / nsamples)
        self.dataset.setArrays(inputs, outputs)