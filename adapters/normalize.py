from dopamine.adapters import Adapter
from numpy import array, inf, ones
from operator import itemgetter
import types

class NormalizingAdapter(Adapter):
    """ This adapter normalizes the states (if they are continuous) between -1 and 1 
        towards the agent. It automatically finds the minimum and maximum values.
    """
    
    # can be applied to all conditions
    inConditions = {}
    
    # define the conditions of the environment
    outConditions = {}    
    
    def __init__(self, normalizeStates=True, scaleActions=False, normalizeRewards=False):
        """ 
            If normalizeStates is given in the Form
                normalizeStates = [(min_0, max_0), (min_1, max_1), ...], 
            the states are assumed to lie within the given boundaries and are scaled down 
            to (-1, 1) per dimension.
            If normalizeStates is set to True, autoscaling is activated (including pretraining)
            and the minimum and maximum of each state dimension is determined automatically
            while running. If normalizeStates is set to False, no scaling of states is performed 
            and the state is passed on to the agent as it comes in.
            
            If scaleActions is given in the form 
                scaleActions = [(min_0, max_0), (min_1, max_1), ...] 
            with one tuple per action dimension, the incoming action is to be assumed 
            between (-1, 1) and scaled between (min_i, max_i) per dimension. If scaleActions
            is False, the actions are passed to the environment as they come in.
            
            If normalizeRewards is set to True, the rewards are normalized between
            (-1, 1) before passed on to the agents. Otherwise, rewards are passed to the
            agent as they come in.
        """
        Adapter.__init__(self)
        
        self.normalizeStates = normalizeStates
        self.normalizeRewards = normalizeRewards                    
        self.scaleActions = scaleActions
        
        # if automatic normalization is activated, require pretraining
        if (type(normalizeStates) == types.BooleanType) and normalizeStates:
            self.autoNormalization = True
            self.requirePretraining = 100
        else:
            self.autoNormalization = False
            self.requirePretraining = 0
            
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
        
        if self.normalizeStates:
            if self.experiment.conditions['discreteStates']:
                self.normalizeStates = False
            else:
                if self.autoNormalization:
                    self.minStates = inf * ones(self.experiment.conditions['stateDim'])
                    self.maxStates = -inf * ones(self.experiment.conditions['stateDim'])
                else:
                    self.minStates = array([tp[0] for tp in self.normalizeStates])
                    self.maxStates = array([tp[1] for tp in self.normalizeStates])
        
        self.minReward = inf
        self.maxReward = -inf   
        
    def applyState(self, state):
        if self.normalizeStates:
            if self.autoNormalization:
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
    
    