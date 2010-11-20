# this class uses the LWPR package from University of Edinburgh
# found at http://www.ipab.inf.ed.ac.uk/slmc/software/lwpr/

from lwpr import LWPR
from numpy import *
from dopamine.agents.valuebased.estimator import Estimator
from dopamine.tools.utilities import one_to_n


class LWPREstimator(Estimator):

    conditions = {'discreteStates':False, 'discreteActions':True}

    def __init__(self, stateDim, actionNum):
        """ initialize with the state dimension and number of actions. """
        self.stateDim = stateDim
        self.actionNum = actionNum
        self.minimum = inf
        self.reset()

    def reset(self):
        # initialize the LWPR function
        self.lwpr = LWPR(self.stateDim + self.actionNum, 1)     
        self.lwpr.init_D = 1.*eye(self.stateDim + self.actionNum)
        self.lwpr.init_alpha = 1.*ones([self.stateDim + self.actionNum, self.stateDim + self.actionNum])
        self.lwpr.meta = True
    
    def train(self):
        pass
        
    def getBestAction(self, state):
        """ returns the action with maximal value in the given state. """
        state = state.flatten()
        return array([argmax([self.getValue(state, array([a])) for a in range(self.actionNum)])])

    def getValue(self, state, action):
        """ returns the value of the given (state,action) pair. """
        state = state.flatten()
        action = action.flatten()
        return self.lwpr.predict(r_[state, one_to_n(action[0], self.actionNum)])

    def updateValue(self, state, action, value):
        self.minimum = min(self.minimum, value)
        # value -= self.minimum
        
        state = state.flatten()
        action = action.flatten()
        self.lwpr.update(r_[state, one_to_n(action, self.actionNum)], array(value).flatten())



class LWPREstimators(Estimator):
    """ This alternative uses several models (one for each action) to 
        predict the values.
    """

    conditions = {'discreteStates':False, 'discreteActions':True}
    trainable = False

    def __init__(self, stateDim, actionNum):
        """ initialize with the state dimension and number of actions. """
        self.stateDim = stateDim
        self.actionNum = actionNum

        # initialize all RBF models, one for each action
        self.reset()

    def getBestAction(self, state):
        """ returns the action with maximal value in the given state. """
        state = state.flatten()
        action = array([argmax([self.getValue(state, array([a])) for a in range(self.actionNum)])])
        return action

    def getValue(self, state, action):
        """ returns the value of the given (state,action) pair. """
        state = state.flatten()
        action = action.flatten()
        return self.models[int(action.item())].predict(state).item()

    def updateValue(self, state, action, value):
        state = state.flatten()
        action = action.flatten()
        self.models[int(action.item())].update(state, array(value).flatten())

    def reset(self):
        """ clear collected training set. """
        # initialize all models, one for each action
        self.models = [LWPR(self.stateDim, 1) for i in range(self.actionNum)]        
        for m in self.models:
            m.init_D = 50*eye(self.stateDim)
            m.init_alpha = 250*ones([self.stateDim, self.stateDim])
            m.meta = True
    
    def train(self):
        pass

    #     """ train individual models for each actions seperately. """
    #     # avoiding the value drift by substracting the minimum of the training set
    #     self.targets = (self.targets - min(self.targets))
    # 
    #     for a in range(self.actionNum):
    #         idx = where(self.actions[:,0] == a)[0]
    #         if len(idx) > 0 and idx.any():
    #             self.models[a].train_ml(self.inputs[idx,:], self.targets[idx,0])
    # 

