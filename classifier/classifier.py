from dopamine.tools.utilities import abstractMethod
from dopamine.tools.dataset import Dataset
import numpy as np

class Classifier(object):
    """ This is the base class for general function approximators. """
    
    parametric = False
    
    def __init__(self, indim, nclasses):
        """ initialize function approximator with input and output dimension. """
        self.indim = indim
        self.nclasses = nclasses
        self.dataset = Dataset(indim, 1)
        self.reset()
    
    def classify(self, inp):
        """ predict the output for the given input. """
        abstractMethod()
        
    def update(self, inp, tgt):
        """ update the function approximator to return something closer
            to target when queried for input next time. Some function
            approximators only collect the input/target tuples here and
            learn only on a call to learn(). """
        self.dataset.append(inp, tgt)
    
    def reset(self):
        """ this initializes the function approximator to an initial state,
            forgetting everything it has learned before. """
        self.dataset.clear()
        
    def train(self):
        """ some function approximators learn offline after collecting all 
            samples. this function executes one such learning step. """
        pass
        
    def _getParameters(self):
        """ getter method for parameters. """
        pass

    def _setParameters(self, parameters):
        """ setter method for parameters. """
        pass

    def _asFlatArray(self, value):
        return np.asarray(value).flatten()
        
        
    # overwrite the two above functions and re-declare this property
    parameters = property(_getParameters, _setParameters)
