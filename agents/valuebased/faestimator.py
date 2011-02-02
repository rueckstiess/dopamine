from numpy import *
from dopamine.agents.valuebased.estimator import Estimator
from dopamine.fapprox import RBF, Linear

from matplotlib import pyplot as plt

class FAEstimator(Estimator):

    conditions = {'discreteStates':False, 'discreteActions':True}
    
    def __init__(self, stateDim, actionNum, faClass=RBF):
        """ initialize with the state dimension and number of actions. """
        self.stateDim = stateDim
        self.actionNum = actionNum
        self.faClass = faClass
        
        # define training and target array
        self.reset()

    def getBestAction(self, state):
        """ returns the action with maximal value in the given state. """
        state = state.flatten()
        action = array([argmax([self.getValue(state, array([a])) for a in range(self.actionNum)])])
        return action

    def getValue(self, state, action):
        """ returns the value of the given (state,action) pair as float. """
        state = state.flatten()
        return self.fas[int(action.item())].predict(state).item()

    def updateValue(self, state, action, value):
        state = state.flatten()
        self.fas[int(action.item())].update(state, value)
   
    def reset(self):
        """ clear collected training set. """
        self.fas = [self.faClass(self.stateDim, 1) for i in range(self.actionNum)]
                
    def train(self):
        """ train individual models for each actions seperately. """
        for a in range(self.actionNum):
            self.fas[a].train()



class OrderedFAEstimator(FAEstimator):
    """ This estimator allows actions to be chosen only once during an episode.
        It will manually assign -inf to the value of an already chosen action,
        making sure the action will not be chosen again, until resetMemory
        is called. The already selected actions have to be specified explicitly
        via rememberAction().
    """

    def __init__(self, stateDim, actionNum, faClass=Linear):
        """ initialize with the state dimension and number of actions. """
        FAEstimator.__init__(self, stateDim, actionNum, faClass)
        self.memory = []

    def getValue(self, state, action):
        """ returns the value of the given (state,action) pair as float. """
        action = int(action.item())
        if action in self.memory:
            return -inf
        else:
            state = state.flatten()
            return self.fas[action].predict(state).item()

    def rememberAction(self, action):
        self.memory.append(int(action.item()))

    def resetMemory(self):
        self.memory = []
