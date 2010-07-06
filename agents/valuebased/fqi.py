from dopamine.agents import Agent
from dopamine.agents.agent import AgentException
from dopamine.agents.valuebased import *
from dopamine.tools.utilities import one_to_n

from numpy import mean, array, r_, c_, atleast_2d, random, equal
import time

class FQIAgent(Agent):
    
    alpha = 0.5
    gamma = 0.9
    
    def __init__(self, estimatorClass=NetworkEstimator):
        """ initialize the agent with the estimatorClass. """
        Agent.__init__(self)
        self.estimatorClass = estimatorClass
    
    def _setup(self, conditions):
        """ if agent is discrete in states and actions create Q-Table. """
        Agent._setup(self, conditions)
        if not (self.conditions['discreteStates'] == False and self.conditions['discreteActions']):
            raise AgentException('FQIAgent expects continuous states and discrete actions. Use adapter or a different environment.')
            
        self.estimator = self.estimatorClass(self.conditions['stateDim'], self.conditions['actionNum'])
    
    def _calculate(self):
        self.action = self.estimator.getBestAction(self.state)
    
    def learn(self):
        """ go through whole episode and make Q-value updates. """  
        for i in range(1):
            if self.estimator.trainable:
                self.estimator._clear()

            for episode in self.history:
                for state, action, reward, nextstate in episode:
                    # don't consider last state
                    if equal(state, nextstate).all():
                        break
                    qvalue = self.estimator.getValue(state, action)
                    bestnext = self.estimator.getValue(nextstate, self.estimator.getBestAction(nextstate))
                    target = (1-self.alpha) * qvalue + self.alpha * (reward + self.gamma * bestnext)

                    self.estimator.updateValue(state, action, target)
            
            if self.estimator.trainable:
                self.estimator._train()

