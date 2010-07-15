from dopamine.agents.direct.controller import Controller
from numpy import *

class LinearController(Controller):
    def __init__(self, stateDim, actionDim):
        """ initializes the controller with the given state and action dimensions."""
        Controller.__init__(self, stateDim, actionDim)
        self.matrix = random.uniform(-0.1, 0.1, (self.stateDim, self.actionDim))
        
    def activate(self, state):
        """ takes the state and returns the associated action. This needs to be 
            overwritten by subclasses."""
        return dot(state, self.matrix)
    
    def _getParameters(self):
        """ getter method for parameters. """
        return self.matrix.flatten()
    
    def _setParameters(self, parameters):
        """ setter method for parameters. """
        self.matrix = parameters.reshape(self.stateDim, self.actionDim)
    
    parameters = property(_getParameters, _setParameters)