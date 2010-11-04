# this class uses the PyBrain library, which can be
# found under http://www.pybrain.org.

from numpy import *
from dopamine.agents.direct.controller import Controller

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers.rprop import RPropMinusTrainer, BackpropTrainer


class NNController(Controller):
    def __init__(self, stateDim, actionDim):
        """ initializes the controller with the given state and action dimensions."""
        Controller.__init__(self, stateDim, actionDim)
        
        # create neural network and pybrain dataset
        self.network = buildNetwork(stateDim, actionDim, bias=False)
        self.dataset = SupervisedDataSet(stateDim, actionDim)
        self.randomize()
        
    def randomize(self):
        """ randomizes the weights of the network. """
        self.network._setParameters(random.normal(0, 0.1, self.network.params.shape))
                 
    def activate(self, state):
        """ takes the state and returns the associated action by calling the
            network's activate() method.
        """
        return self.network.activate(state)
    

    def paramsDerivative(self, state, derivs):
        """ this function receive the derivatives of the actions with respect
            to the mean and needs to return the derivatives with respect to the 
            parameters. 
        """
        self.network.reset()
        self.network.activate(state)
        self.network.backActivate(derivs)
        return self.network.derivs.flatten()

    def _getParameters(self):
        """ getter method for parameters. """
        return self.network.params
    
    def _setParameters(self, parameters):
        """ setter method for parameters. """
        self.network._setParameters(parameters)
    
    parameters = property(_getParameters, _setParameters)