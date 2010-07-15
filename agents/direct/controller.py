from dopamine.tools.utilities import abstractMethod

class Controller(object):
    def __init__(self, stateDim, actionDim):
        """ initializes the controller with the given state and action dimensions."""
        self.stateDim = stateDim
        self.actionDim = actionDim
        
    def activate(self, state):
        """ takes the state and returns the associated action. This needs to be 
            overwritten by subclasses."""
        abstractMethod()
    
    def randomize(self):
        pass
        
    def _getParameters(self):
        """ getter method for parameters. """
        pass
    
    def _setParameters(self, parameters):
        """ setter method for parameters. """
        pass
    
    # overwrite the two above functions and re-declare this property
    parameters = property(_getParameters, _setParameters)
    