class ExperimentException(Exception):
    pass

class Experiment(object):
    
    def __init__(self, environment, agent):
        self.environment = environment
        self.agent = agent
        self.adapters_ = []
        
        # marker that stores if setup on agent has been completed
        self.setupComplete = False
            
    def _flattenConditions(self):
        """ flattens the environment and adapters conditions to one conditions dictionary. """
        conditions = dict([(k,v) for k,v in self.environment.conditions.items()])
        
        for a in self.adapters_:
            for k,v in a.outConditions.items():
                conditions[k] = v
                
        return conditions
    
    @property
    def conditions(self):
        return self._flattenConditions()
    
    @property
    def adapters(self):
        return self.adapters_
        
    def addAdapter(self, adapter):
        """ add an adapter to the end of the adapter list if it is compatible. """
        # get the flattened conditions dictionary
        conditions = self.conditions

        # check for each inCondition if it is compatible to the adapter stack
        for c,v in adapter.inConditions.items():
            if not c in conditions:
                # condition could not be found in adapter stack or environment
                raise ExperimentException('condition "%s" could not be found in adapter stack or environment.'%c)
            
            if conditions[c] != v:
                # condition could be found but does not match
                raise ExperimentException('condition "%s" is not compatible to previous environment/adapter. Value must be %s'%(c, conditions[c]))
        
        # every condition matches, set experiment and add to adapter stack
        adapter.setExperiment(self)
        self.adapters_.append(adapter)
        
        # conditions have changed, new agent setup is necessary
        self.setupComplete = False
    
    
    def interact(self):
        """ run one interaction between agent and environment. The state from
            the environment gets passed through all registered adapters before
            it is presented to the agent. The agent returns an action, which
            goes again through all adapters (in reverse order) and is then 
            executed in the environment. A reward is then passed through the
            adapters to the agent.
        """
        if not self.setupComplete:
            self.agent._setup(self.conditions)
            self.setupComplete = True
            
        state = self.environment.getState()
        for adapter in self.adapters_:
            state = adapter.applyState(state)
        self.agent.integrateState(state)
        
        action = self.agent.getAction()
        for adapter in reversed(self.adapters_):
            action = adapter.applyAction(action)
        self.environment.performAction(action)
        
        reward = self.environment.getReward()
        for adapter in self.adapters_:
            reward = adapter.applyReward(reward)
        self.agent.giveReward(reward)
    
    def runEpisode(self, reset=True):
        """ resets the environment if reset is True, then calls interact() until 
            the environment and its adapters signals the end of an episode. 
        """
        if not self.conditions['episodic']:
            raise ExperimentException('Environment is not episodic, or adapters transformed it into non-episodic. Use interact() method.')
        
        # reset adapters and environment
        if reset:
            for adapter in self.adapters_:
                adapter.reset()
            self.environment.reset()
        
        # get environment's episodeFinished and push it through all adapters
        episodeFinished = self.environment.episodeFinished()
        for adapter in self.adapters_:
            episodeFinished = adapter.applyEpisodeFinished(episodeFinished)
        
        # while the resulting episodeFinished is False, loop over interactions
        while not episodeFinished:
            self.interact()
            episodeFinished = self.environment.episodeFinished()
            for adapter in self.adapters_:
                episodeFinished = adapter.applyEpisodeFinished(episodeFinished)
        
        # tell agent that it should start a new episode
        self.agent.newEpisode()
        