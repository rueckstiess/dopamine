from rllib.agents import Agent
from rllib.agents.agent import AgentException
from rllib.agents.valuebased import NetworkEstimator
from rllib.tools.utilities import one_to_n

from numpy import mean, array, r_, c_, atleast_2d
import time

class NFQAgent(Agent):
    
    alpha = 0.1
    gamma = 0.9
    
    def _setup(self, conditions):
        """ if agent is discrete in states and actions create Q-Table. """
        Agent._setup(self, conditions)
        if not (self.conditions['discreteStates'] == False and self.conditions['discreteActions']):
            raise AgentException('QAgent expects discrete states and actions. Use adapter or a different environment.')
            
        self.estimator = NetworkEstimator(self.conditions['stateDim'], self.conditions['actionNum'])
    
    def _calculate(self):
        self.action = self.estimator.getMaxAction(self.state)
    
    def learn(self):
        """ go through whole episode and make Q-value updates. """  
        for i in range(1):
            self.estimator.dataset.clear()
          
            for episode in self.history:
                for state, action, reward, nextstate in episode:
                    qvalue = self.estimator.getValue(state, action)
                    maxnext = self.estimator.getValue(nextstate, self.estimator.getMaxAction(nextstate))
                    target = qvalue + self.alpha * (reward + self.gamma * maxnext - qvalue)

                    self.estimator.dataset.addSample(r_[state, one_to_n(action, self.conditions['actionNum'])], target)
            self.estimator.train(30)

