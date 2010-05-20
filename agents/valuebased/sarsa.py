from dopamine.agents import QAgent

class SARSAAgent(Agent):
    
    alpha = 0.5
    gamma = 0.9
    
    def learn(self):
        """ go through whole episode and make Q-value updates. """
        for episode in self.history:
            for i, (state, action, reward, nextstate) in enumerate(episode):
                if i+1 >= len(episode):
                    break
                    
                _, nextaction, _ = episode[i+1]
            
                state = int(state)
                action = int(action)
                nextstate = int(nextstate)
                nextaction = int(nextaction)
     
                qvalue = self.estimator.getValue(self.state, self.action)
                nextval = self.estimator.getValue(nextstate, nextaction)
                
                self.estimator.updateValue(state, action, qvalue + self.alpha * (reward + self.gamma * nextval - qvalue))

