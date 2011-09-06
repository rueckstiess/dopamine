from dopamine.agents.valuebased.q import QAgent

class SARSAAgent(QAgent):
    
    alpha = 0.5
    gamma = 0.9
    
    def learn(self):
        """ go through whole episode and make Q-value updates. """
        for episode in self.history:
            for i, (state, action, reward, nextstate) in enumerate(episode):
                state = int(state)
                action = int(action)
     
                qvalue = self.estimator.getValue(self.state, self.action)
                if nextstate != None:
                    _, nextaction, _ = episode[i+1]
                    nextval = self.estimator.getValue(int(nextstate), int(nextaction))
                else:
                    nextval = 0.
                
                self.estimator.updateValue(state, action, qvalue + self.alpha * (reward + self.gamma * nextval - qvalue))

