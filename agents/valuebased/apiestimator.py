from numpy import *
from random import choice
from dopamine.agents.valuebased.estimator import Estimator
from dopamine.fapprox import Linear

from matplotlib import pyplot as plt

class APIEstimator(Estimator):

    conditions = {'discreteStates':False, 'discreteActions':True}
    
    def __init__(self, stateDim, actionNum):
        """ initialize with the state dimension and number of actions. """
        self.stateDim = stateDim
        self.actionNum = actionNum
                
        # define training and target array
        self.reset()

    def getBestAction(self, state):
        """ returns the action with maximal value in the given state. if several
            actions have the same value, pick one at random.
        """
        state = state.flatten()
        values = array([self.getValue(state, array([a])) for a in range(self.actionNum)])
        maxvalues = where(values == values.max())[0]
        if len(maxvalues) > 0:
            action = array([choice(maxvalues)])
        else:
            # this should not happen, but it does in rare cases, return the first action
            action = array([0])
        return action

    
    def _vectorBlockState(self, state, action):
        state = state.flatten()
        action = action.item()
        block = zeros(self.stateDim*self.actionNum)
        block[self.stateDim*action:self.stateDim*(action+1)] = state
        return block


    def getValue(self, state, action):
        """ returns the value of the given (state,action) pair as float. """
        return self.fa.predict(self._vectorBlockState(state, action)).item()

    def updateValue(self, state, action, value):
        self.fa.update(self._vectorBlockState(state, action), value)
   
    def reset(self):
        """ clear collected training set. """
        # special case to clean up lwpr models that were pickled
        if self.faClass == LWPRFA:
            fa._cleanup()
        self.fa = Linear(self.stateDim * self.actionNum, 1)
                
    def train(self):
        """ train individual models for each actions seperately. """
        self.fa.train()