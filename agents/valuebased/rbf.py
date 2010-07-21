from numpy import *
from dopamine.agents.valuebased.estimator import Estimator
from dopamine.tools.rbf import RBF

class RBFEstimator(Estimator):

    conditions = {'discreteStates':False, 'discreteActions':True}
    trainable = True
    
    def __init__(self, stateDim, actionNum):
        """ initialize with the state dimension and number of actions. """
        self.stateDim = stateDim
        self.actionNum = actionNum
        
        # define training and target array
        self.inputs = zeros((0,stateDim))
        self.actions = zeros((0, 1))
        self.targets = zeros((0, 1))
        
        # initialize all RBF models, one for each action
        self.models = [RBF(stateDim, 20, 1) for i in range(actionNum)]

    def getBestAction(self, state):
        """ returns the action with maximal value in the given state. """
        state = state.flatten()
        action = array([argmax([self.getValue(state, array([a])) for a in range(self.actionNum)])])
        return action

    def getValue(self, state, action):
        """ returns the value of the given (state,action) pair. """
        state = state.flatten()
        action = action.flatten()
        return self.models[int(action.item())].test(state.reshape(1, self.stateDim)).item()

    def updateValue(self, state, action, value):
        self.inputs = r_[self.inputs, state.reshape(1, self.stateDim)]
        self.actions = r_[self.actions, action.reshape(1, 1)]
        self.targets = r_[self.targets, asarray(value).reshape(1, 1)]
   
    def _clear(self):
        """ clear collected training set. """
        self.inputs = zeros((0, self.stateDim))
        self.actions = zeros((0, 1))
        self.targets = zeros((0, 1))
                
    def _train(self):
        """ train individual models for each actions seperately. """
        if len(self.targets) == 0:
            return
            
        # avoiding the value drift by substracting the minimum of the training set
        self.targets = (self.targets - min(self.targets))
        
        for a in range(self.actionNum):
            idx = where(self.actions[:,0] == a)[0]
            if len(idx) > 0 and idx.any():
                self.models[a].train_ml(self.inputs[idx,:], self.targets[idx,0])
     

class RBFOnlineEstimator(RBFEstimator):

    conditions = {'discreteStates':False, 'discreteActions':True}
    trainable = False
    
    def __init__(self, stateDim, actionNum):
        """ initialize with the state dimension and number of actions. """
        self.stateDim = stateDim
        self.actionNum = actionNum
        
        self.minimum = inf
        
        # initialize all RBF models, one for each action
        self.models = [RBF(stateDim, 20, 1) for i in range(actionNum)]
        
    def updateValue(self, state, action, value):
        self.minimum = min(self.minimum, value)
        value -= self.minimum
        
        self.models[action.item()].add_sample_map(state.reshape(1, self.stateDim), asarray(value).reshape(1, 1))
        