from rllib.tools.utilities import abstractMethod

class Estimator(object):
    
    discreteStates = False
    discreteActions = False
    
    def getMaxAction(self, state):
        """ returns the action with the highest value in the given state. """
        abstractMethod()
    
    def getValue(self, state, action):
        """ returns the value of the (state, action) tuple. """
        abstractMethod()