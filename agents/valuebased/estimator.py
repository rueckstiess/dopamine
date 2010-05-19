from dopamine.tools.utilities import abstractMethod

class Estimator(object):
    
    conditions = {'discreteStates':False, 'discreteActions':False}
    trainable = False
    
    def getBestAction(self, state):
        """ returns the action with the highest value in the given state. """
        abstractMethod()
    
    def getValue(self, state, action):
        """ returns the value of the (state, action) tuple. """
        abstractMethod()
    
    def updateValue(self, state, action, value):
        """ updates the value for the given (state, action) tuple. """
        pass
    
    def _train(self):
        pass