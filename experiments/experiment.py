class Experiment(object):
    
    def __init__(self, environment, agent):
        self.environment = environment
        self.agent = agent
        self.adapters = []
        
    def addAdapter(self, adapter):
        """ add an adapter to the end of the adapter list. """
        self.adapters.append(adapter)
    
    def interact(self):
        """ run one interaction between agent and environment. The state from
            the environment gets passed through all registered adapters before
            it is presented to the agent. The agent returns an action, which
            goes again through all adapters (in reverse order) and is then 
            executed in the environment. A reward is then passed through the
            adapters to the agent.
        """
        state = self.environment.getState()
        for adapter in self.adapters:
            state = adapter.applyState(state)
        self.agent.integrateObservation(state)
        
        action = self.agent.getAction()
        for adapter in reversed(self.adapters):
            action = adapter.applyAction(action)
        self.environment.performAction(action)
        
        reward = self.environment.getReward()
        for adapter in self.adapters:
            reward = adapter.applyReward(reward)
        self.agent.giveReward(reward)
    
    def runEpisode(self):
        """ resets the environment, then calls interact() until the environment 
            signals the end of an episode. 
        """
        self.environment.reset()
        while not self.environment.episodeFinished():
            self.interact()
        