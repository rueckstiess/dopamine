from rllib.agents import Agent
from rllib.agents.valuebased import Table

class QAgent(Agent):
    pass
    
    # def __init__(self):
    #     
    #     
    # def learn(self):
    #     for episode in self.history:
    #         for state, action, reward, nextstate in episode:
    #         
    #             state = int(state)
    #             action = int(action)
    #             nextstate = int(nextstate)
    #  
    #             qvalue = self.module.getValue(self.laststate, self.lastaction)
    #             maxnext = self.module.getValue(state, self.module.getMaxAction(state))
    #             self.module.updateValue(self.laststate, self.lastaction, qvalue + self.alpha * (reward + self.gamma * maxnext - qvalue))
    # 
    #             # move state to oldstate
    #             self.laststate = state
    #             self.lastaction = action
    