from numpy import *
from dopamine.tools.utilities import one_to_n
from dopamine.agents.valuebased.estimator import Estimator

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers.rprop import RPropMinusTrainer, BackpropTrainer

class NetworkEstimator(Estimator):

    conditions = {'discreteStates':False, 'discreteActions':True}
    trainable = True
    
    def __init__(self, stateDim, actionNum):
        """ initialize with the state dimension and number of actions. """
        self.stateDim = stateDim
        self.actionNum = actionNum
        self.network = buildNetwork(stateDim + actionNum, (stateDim + actionNum), 1)
        self.network._setParameters(random.normal(0, 0.1, self.network.params.shape))
        self.dataset = SupervisedDataSet(stateDim + actionNum, 1)

    def getBestAction(self, state):
        """ returns the action with maximal value in the given state. """
        return array([argmax([self.getValue(state, array([a])) for a in range(self.actionNum)])])

    def getValue(self, state, action):
        """ returns the value of the given (state,action) pair. """
        return self.network.activate(r_[state, one_to_n(action[0], self.actionNum)])

    def updateValue(self, state, action, value):
        self.dataset.addSample(r_[state, one_to_n(action, self.actionNum)], value)

    def _train(self):
        # train module with backprop/rprop on dataset
        trainer = RPropMinusTrainer(self.network, dataset=self.dataset, batchlearning=True, verbose=False)
        # trainer = BackpropTrainer(self.network, dataset=self.dataset, batchlearning=True, verbose=True)
        trainer.trainEpochs(100)
     
        

