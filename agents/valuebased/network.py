from numpy import *
from rllib.tools.utilities import one_to_n
from rllib.agents.valuebased.estimator import Estimator
from rllib.tools.ffn import FeedForwardNetwork

class NetworkEstimator(Estimator):

    conditions = {'discreteStates':False, 'discreteActions':True}

    def __init__(self, stateDim, actionNum):
        """ initialize with the state dimension and number of actions. """
        self.stateDim = stateDim
        self.actionNum = actionNum
        self.network = FeedForwardNetwork(stateDim + actionNum, (stateDim + actionNum) * 3, 1)

    def getMaxAction(self, state):
        """ returns the action with maximal value in the given state. """
        return array([argmax([self.getValue(state, array([a])) for a in range(self.actionNum)])])

    def getValue(self, state, action):
        """ returns the value of the given (state,action) pair. """
        self.network.forward(r_[state, one_to_n(action[0], self.actionNum)])
        return self.network.getOutput()

    def train(self, state, action, target):
        self.getValue(state, action)
        return self.network.backward(target)

