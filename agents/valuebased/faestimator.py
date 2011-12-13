from numpy import *
from random import choice
from dopamine.agents.valuebased.estimator import Estimator
from dopamine.fapprox import RBF, LWPRFA

from matplotlib import pyplot as plt

class FAEstimator(Estimator):

    conditions = {'discreteStates':False, 'discreteActions':True}
    
    def __init__(self, stateDim, actionNum, faClass=RBF, ordered=False):
        """ initialize with the state dimension and number of actions. """
        self.stateDim = stateDim
        self.actionNum = actionNum
        self.faClass = faClass
        self.ordered = ordered
        self.fas = []
                
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
        values = array([self.getValue(state, array([a])) for a in range(self.actionNum)])
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
        return self.fas[action].predict(state).item()


    def updateValue(self, state, action, value):
        state = state.flatten()
        self.fas[int(action.item())].update(state, value)
   
    def reset(self):
        """ clear collected training set. """
        # special case to clean up lwpr models that were pickled
        if self.faClass == LWPRFA:
            for fa in self.fas:
                fa._cleanup()
        self.fas = [self.faClass(self.stateDim, 1) for i in range(self.actionNum)]
                
    def train(self):
        """ train individual models for each actions seperately. """
        for a in range(self.actionNum):
            self.fas[a].train()

    def rememberAction(self, action):
        if self.ordered:
            self.memory.append(int(action.item()))

    def resetMemory(self):
        if self.ordered:
            self.memory = []

