from numpy import *
from random import choice
from dopamine.agents.valuebased.faestimator import FAEstimator
from dopamine.fapprox import Linear, LWPRFA
import types

from matplotlib import pyplot as plt

class VectorBlockEstimator(FAEstimator):

    conditions = {'discreteStates':False, 'discreteActions':True}
    
    def __init__(self, stateDim, actionNum, faClass=Linear, ordered=False):
        self.fa = None
        FAEstimator.__init__(self, stateDim, actionNum, faClass, ordered)

    def _vectorBlockState(self, state, action):
        state = state.flatten()
        if type(action) != types.IntType:
            action = action.item()
        block = zeros(self.stateDim*self.actionNum)
        block[self.stateDim*action:self.stateDim*(action+1)] = state
        return block

    def getValue(self, state, action):
        """ returns the value of the given (state,action) pair as float. """
        action = int(action.item())

        if self.ordered and (action in self.memory):
            return -inf

        return self.fa.predict(self._vectorBlockState(state, action)).item()

    def updateValue(self, state, action, value):
        self.fa.update(self._vectorBlockState(state, action), value)
   
    def reset(self):
        """ clear collected training set. """
        # special case to clean up lwpr models that were pickled
        if self.faClass == LWPRFA:
            if self.fa:
                self.fa._cleanup()

        self.fa = self.faClass(self.stateDim * self.actionNum, 1)
                
    def train(self):
        """ train individual models for each actions seperately. """
        self.fa.train()

