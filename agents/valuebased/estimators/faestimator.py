from numpy import *
from dopamine.agents.valuebased.estimator import Estimator
from dopamine.fa import RBF

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
        self.models = [self.faClass(self.stateDim, 1) for i in range(self.actionNum)]
                
    def train(self):
        """ train individual models for each actions seperately. """
        if len(self.targets) == 0:
            return
            
        # avoiding the value drift by substracting the minimum of the training set
        for a in range(self.actionNum):
            self.fas[a].learn()
     
