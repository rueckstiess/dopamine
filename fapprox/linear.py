from dopamine.fapprox.fa import FA
import numpy as np

class Linear(FA):
    
    parametric = True
        
    def predict(self, inp):
        """ predict the output for the given input. """
        inp = self._asFlatArray(inp)
        return self._asFlatArray(np.dot(np.r_[inp, np.array([1])], self.matrix))

    def reset(self):
        """ this initializes the function approximator to an initial state,
            forgetting everything it has learned before. """
        FA.reset(self)
        self.matrix = np.random.uniform(-0.1, 0.1, (self.indim + 1, self.outdim))

    def train(self):
        """ some function approximators learn offline after collecting all 
            samples. this function executes one such learning step. """
        if len(self.dataset) == 0:
            return
        self.matrix = np.dot(np.linalg.pinv(np.c_[self.dataset.inputs, np.ones((len(self.dataset), 1))]), self.dataset.targets)

    def _getParameters(self):
        """ getter method for parameters. """
        return self.matrix.flatten()

    def _setParameters(self, parameters):
        """ setter method for parameters. """
        self.matrix = parameters.reshape(self.indim, self.outdim)
        
    parameters = property(_getParameters, _setParameters)
            