from dopamine.adapters import Adapter
from numpy import array, inf, ones

class NormalizingAdapter(Adapter):
    """ This adapter normalizes the states (if they are continuous) between -1 and 1 
        towards the agent. It automatically finds the minimum and maximum values.
    """
    
    # can be applied to all conditions
    inConditions = {}
    
    # define the conditions of the environment
    outConditions = {}    
    
    def __init__(self, normalizeRewards=False):
        self.normalizeRewards = normalizeRewards
    
    def setExperiment(self, experiment):
        Adapter.setExperiment(self, experiment)
        
        if not self.experiment.conditions['discreteStates']:
            self.minStates = inf * ones(self.experiment.conditions['stateDim'])
            self.maxStates = -inf * ones(self.experiment.conditions['stateDim'])
        
        self.minReward = inf
        self.maxReward = -inf   
        
    def applyState(self, state):
        if not self.experiment.conditions['discreteStates']:
            self.minStates = array([min(a, b) for a,b in zip(self.minStates, state)])
            self.maxStates = array([max(a, b) for a,b in zip(self.maxStates, state)])
            denominator = self.maxStates - self.minStates
            if denominator.all():
                state = (state - self.minStates) / denominator * 2. - 1.
        return state
    
    def applyReward(self, reward):
        if self.normalizeRewards:
            self.minReward = min(self.minReward, reward)
            self.maxReward = max(self.maxReward, reward)
            denominator = self.maxReward - self.minReward
            if denominator != 0:
                reward = (reward - self.minReward) / denominator * 2. - 1.
        return reward
    
    