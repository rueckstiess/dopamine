from rllib.agents import Agent
from rllib.agents.agent import AgentException
from rllib.agents.valuebased import NetworkEstimator

from numpy import mean

class NFQAgent(Agent):
    
    alpha = 0.5
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
        errors = []
        for episode in self.history:
            for state, action, reward, nextstate in episode:
     
                qvalue = self.estimator.getValue(self.state, self.action)
                maxnext = self.estimator.getValue(nextstate, self.estimator.getMaxAction(nextstate))
                tgt = qvalue + self.alpha * (reward + self.gamma * maxnext - qvalue)
                errors.append(self.estimator.train(state, action, tgt))
        print mean(errors)

