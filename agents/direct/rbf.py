from numpy import *
from dopamine.agents.direct.controller import Controller
from dopamine.tools.rbf import RBF

class RBFController(Controller):
    def __init__(self, stateDim, actionDim):
        """ initializes the controller with the given state and action dimensions."""
        Controller.__init__(self, stateDim, actionDim)
        
        self.numCenters = 10
        # create neural network and pybrain dataset
        self.model = RBF(stateDim, self.numCenters, actionDim)
        self.model.beta = 6.
        
    def randomize(self):
        """ randomizes the weights of the network. """
        self.parameters = random.random((self.numCenters, self.actionDim))
               
    def activate(self, state):
        """ takes the state and returns the associated action by calling the
            network's activate() method.
        """
        return self.model.test(state.reshape(1, self.stateDim))

    def getDerivative(self, state, derivs):
        """ this function receive the derivatives of the actions with respect
            to the mean and needs to return the derivatives with respect to the 
            parameters. 
        """
        G = self.model._designMatrix(asarray(state).reshape(1, self.stateDim))
        return dot(G.T, asarray(derivs).reshape(1, self.actionDim)).flatten()

    def _getParameters(self):
        """ getter method for parameters. """
        return self.model.W.flatten()
    
    def _setParameters(self, parameters):
        """ setter method for parameters. """
        self.model.W = parameters.reshape(self.numCenters, self.actionDim)
    
    parameters = property(_getParameters, _setParameters)