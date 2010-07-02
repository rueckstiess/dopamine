# this class uses the LWPR package from University of Edinburgh
# found at http://www.ipab.inf.ed.ac.uk/slmc/software/lwpr/

from numpy import *
from lwpr import *
from dopamine.agents.valuebased.estimator import Estimator


class LWPREstimator(Estimator):
    
    conditions = {'discreteStates':False, 'discreteActions':True}
    trainable = False
    
    def __init__(self, stateDim, actionNum):
        """ initialize with the state dimension and number of actions. """
        self.stateDim = stateDim
        self.actionNum = actionNum
        
        # initialize all RBF models, one for each action
        self.models = [LWPR(stateDim, 1) for i in range(actionNum)]

    def getBestAction(self, state):
        """ returns the action with maximal value in the given state. """
        state = state.flatten()
        action = array([argmax([self.getValue(state, array([a])) for a in range(self.actionNum)])])
        return action

    def getValue(self, state, action):
        """ returns the value of the given (state,action) pair. """
        state = state.flatten()
        action = action.flatten()
        return self.models[int(action.item())].predict(state.reshape(1, self.stateDim)).item()

    def updateValue(self, state, action, value):
        action = action.flatten()
        self.models[int(action.item())].update(state, asarray(value))

    def _clear(self):
        """ clear collected training set. """
        # initialize all RBF models, one for each action
        self.models = [LWPR(stateDim, 1) for i in range(actionNum)]        
    
    # def _train(self):
    #     """ train individual models for each actions seperately. """
    #     # avoiding the value drift by substracting the minimum of the training set
    #     self.targets = (self.targets - min(self.targets))
    # 
    #     for a in range(self.actionNum):
    #         idx = where(self.actions[:,0] == a)[0]
    #         if len(idx) > 0 and idx.any():
    #             self.models[a].train_ml(self.inputs[idx,:], self.targets[idx,0])
    # 

