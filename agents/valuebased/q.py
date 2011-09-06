from dopamine.agents.agent import Agent, AgentException
from dopamine.agents.valuebased.table import TableEstimator

class QAgent(Agent):
    
    alpha = 0.5
    gamma = 0.9
    
    def _setup(self, conditions):
        """ if agent is discrete in states and actions create Q-Table. """
        Agent._setup(self, conditions)
        if not (self.conditions['discreteStates'] and self.conditions['discreteActions']):
            raise AgentException('QAgent expects discrete states and actions. Use adapter or a different environment.')
            
        self.estimator = TableEstimator(self.conditions['stateNum'], self.conditions['actionNum'])
    
    def _calculate(self):
        self.action = self.estimator.getBestAction(self.state)
    
    def learn(self):
        """ go through whole episode and make Q-value updates. """
        for episode in self.history:
            for state, action, reward, nextstate in episode:
            
                state = int(state)
                action = int(action)
     
                qvalue = self.estimator.getValue(self.state, self.action)
                if nextstate != None:
                    maxnext = self.estimator.getValue(int(nextstate), self.estimator.getBestAction(nextstate))
                else:
                    maxnext = 0.

                self.estimator.updateValue(state, action, qvalue + self.alpha * (reward + self.gamma * maxnext - qvalue))

    