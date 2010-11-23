from dopamine.agents.valuebased.estimator import Estimator
from dopamine.tools.knn import KNN

import numpy as np
from operator import itemgetter
from bisect import insort

class KNNEstimator(Estimator):
    
    conditions = {'discreteStates':False, 'discreteActions':True}
    
    def __init__(self, stateDim, actionNum):
        self.stateDim = stateDim
        self.actionNum = actionNum
        
        self.reset()
    
    def getBestAction(self, state):
        """ returns the action with the highest value in the given state. """
        state = state.flatten()
        action = np.asarray([np.argmax([self.getValue(state, np.asarray([a])) for a in range(self.actionNum)])])
        return action
    
    def getValue(self, state, action):
        """ returns the value of the (state, action) tuple. """
        return self.models[int(action.item())].predict(state)
    
    def updateValue(self, state, action, value):
        """ updates the value for the given (state, action) tuple. """
        self.models[int(action.item())].addPoint(state, value)
    
    def reset(self):
        """ for estimators that collect samples and require training,
            this function should clear the entire dataset and reset 
            the estimator.
        """
        self.models = [KNN() for _ in range(self.actionNum)]

    def train(self):
        """ for estimators that collect samples and require training,
            this function should run one training step.
        """
        pass