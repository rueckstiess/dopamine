from rllib.agents import Agent
from rllib.agents.agent import AgentException
from rllib.agents.valuebased import Table

class QAgent(Agent):
    
    def setup(self, discreteStates, discreteActions, states, actions):
        """ if agent is discrete in states and actions create Q-Table. """
        Agent.setup(self, discreteStates, discreteActions, states, actions)
        if not (discreteStates and discreteActions):
            raise AgentException('QAgent expects discrete states and actions. Use adapter or a different environment.')
            
        self.table = Table(self.stateNum, self.actionNum)
        
    
    def learn(self):
        """ go through whole episode and make Q-value updates. """
        for episode in self.history:
            for state, action, reward, nextstate in episode:
            
                state = int(state)
                action = int(action)
                nextstate = int(nextstate)
     
                qvalue = self.table.getValue(self.state, self.action)
                maxnext = self.table.getValue(nextstate, self.table.getMaxAction(nextstate))
                
                self.table.updateValue(state, action, qvalue + self.alpha * (reward + self.gamma * maxnext - qvalue))

    