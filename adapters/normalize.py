from rllib.adapters import Adapter
from numpy import array, inf, ones

class NormalizingAdapter(Adapter):
    """ This adapter normalizes the states and actions (if they are continuous)
        between -1 and 1 towards the agent.
    """
    
    # can be applied to all conditions
    inConditions = {}
    
    # define the conditions of the environment
    outConditions = {}    
    
    def setExperiment(self, experiment):
        Adapter.setExperiment(self, experiment)
        
        if not self.experiment.conditions['discreteStates']:
            self.minStates = inf * ones(self.experiment.conditions['stateDim'])
            self.maxStates = -inf * ones(self.experiment.conditions['stateDim'])
        
        if not self.experiment.conditions['discreteActions']:
            self.minActions = inf * ones(self.experiment.conditions['actionDim'])
            self.maxActions = -inf * ones(self.experiment.conditions['actionDim'])
        
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
        
    def applyAction(self, action):
        if not self.experiment.conditions['discreteActions']:
            self.minActions = array([min(a, b) for a,b in zip(self.minActions, action)])
            self.maxActions = array([max(a, b) for a,b in zip(self.maxActions, action)])
            denominator = self.maxActions - self.minActions
            if denominator.all():
                action = (action + 1.0) / 2 * (self.maxActions - self.minActions) + self.minActions
        return action
    
    def applyReward(self, reward):
        self.minReward = min(self.minReward, reward)
        self.maxReward = max(self.maxReward, reward)
        denominator = self.maxReward - self.minReward
        if denominator != 0:
            reward = (reward - self.minReward) / denominator * 2. - 1.
        return reward
          
    