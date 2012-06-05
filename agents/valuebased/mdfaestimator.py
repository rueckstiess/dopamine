from numpy import *
from random import choice
from dopamine.agents.valuebased.estimator import Estimator
from dopamine.fapprox import Linear, LWPRFA

from matplotlib import pyplot as plt

class MDFAEstimator(Estimator):

    conditions = {'discreteStates':False, 'discreteActions':True}
    
    def __init__(self, stateDim, actionNum, faClass=Linear, ordered=False):
        """ initialize with the state dimension and number of actions. """
        self.stateDim = stateDim
        self.actionNum = actionNum
        self.faClass = faClass
        self.ordered = ordered
        self.fa = None
                
        # create memory for ordered estimator
        if self.ordered:
            self.memory = []

        # define training and target array
        self.reset()

    def getBestAction(self, state):
        """ returns the action with maximal value in the given state. if several
            actions have the same value, pick one at random.
        """
        state = state.flatten()
        values = self.fa.predict(state)
        maxvalues = where(values == values.max())[0]
        if len(maxvalues) > 0:
            action = array([choice(maxvalues)])
        else:
            # this should not happen, but it does in rare cases, return the first action
            action = array([0])
        return action

    def getValue(self, state, action):
        """ returns the value of the given (state,action) pair as float. """
        action = int(action.item())

        if self.ordered and (action in self.memory):
            return -inf
        
        state = state.flatten()
        return self.fa.predict(state)[action]

    def updateValue(self, state, action, value):
        action = int(action.item())
        state = state.flatten()
        output = self.fa.predict(state)
        output[action] = value
        self.fa.update(state, output)
   
    def reset(self):
        """ clear collected training set. """
        # special case to clean up lwpr models that were pickled
        if self.faClass == LWPRFA:
            fa._cleanup()
        self.fa = self.faClass(self.stateDim, self.actionNum)
                
    def train(self):
        """ train individual models for each actions seperately. """
        self.fa.train()

    def rememberAction(self, action):
        if self.ordered:
            self.memory.append(int(action.item()))

    def resetMemory(self):
        if self.ordered:
            self.memory = []

