from dopamine.adapters import Adapter
from numpy import array, inf, ones
from operator import itemgetter

class NormalizingAdapter(Adapter):
    """ This adapter normalizes the states (if they are continuous) between -1 and 1 
        towards the agent. It automatically finds the minimum and maximum values.
    """
    
    # can be applied to all conditions
    inConditions = {}
    
    # define the conditions of the environment
    outConditions = {}    
    
    def __init__(self, scaleActions=False, normalizeRewards=False):
        Adapter.__init__(self)
        
        self.normalizeRewards = normalizeRewards                    
        self.scaleActions = scaleActions
            
    def setExperiment(self, experiment):
        Adapter.setExperiment(self, experiment)
        
        if self.scaleActions:
            if self.experiment.conditions['discreteActions']:
                self.scaleActions = False
            else:
                if len(self.scaleActions) != self.experiment.conditions['actionDim']:
                    # TODO: use less generic exception, e.g. AdapterException
                    raise SystemExit('scaleActions must contain a pair of min/max values for each action dimension (%i). Only %i are given.'%(self.inConditions['actionDim'], len(scaleActions)))
                for p in self.scaleActions:
                    if len(p) != 2:
                        raise SystemExit('scaleActions must contain a pair of min/max values for each action dimension. %s is not a pair. '%str(p))
        
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

    def applyAction(self, action):
        """ assumes that the given action is between -1 and 1. scales the action so 
            it will be between min_i, max_i for each dimension, specified by the tuple
            scaleActions = [(min_0, max_0), (min_1, max_1), ...] in scaleActions.
        """ 
        if self.scaleActions:
            minvec = array(map(itemgetter(0), self.scaleActions))
            maxvec = array(map(itemgetter(1), self.scaleActions))
            action = (action + 1) / 2 * (maxvec - minvec) + minvec
        return action
    
    def applyReward(self, reward):
        if self.normalizeRewards:
            self.minReward = min(self.minReward, reward)
            self.maxReward = max(self.maxReward, reward)
            denominator = self.maxReward - self.minReward
            if denominator != 0:
                reward = (reward - self.minReward) / denominator * 2. - 1.
        return reward
    
    