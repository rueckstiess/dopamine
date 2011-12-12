from dopamine.experiments.experiment import Experiment


class APIExperiment(Experiment):

    def runEpisode(self):
        """ resets the environment in a random state, then creates rollouts until
            the environment and its adapters signals the end of an episode, for each
            possible action from the initial state. Note: this call will generate
            numAction rollouts, not just one.  
        """
        if not self.conditions['episodic']:
            raise ExperimentException('Environment is not episodic, or adapters transformed it into non-episodic. Use interact() method.')
        
        if not self.environment.generator:
            raise ExperimentException('Environment is not a generator and cannot be used to start in random states.')

        if not self.conditions['discreteActions']:
            raise ExperimentException('API Experiments require a discrete action space, but the environment has a continuous action space.')

        # reset adapters and environment
        for adapter in self.adapters_:
            adapter.reset()
        
        # pick a random state from the environment
        randomState = self.environment.getRandomState()
        
        # for each possible action, run one Rollout
        for action in range(self.conditions['actionNum']):
            self.environment.resetToState(randomState)
        
            # get environment's episodeFinished and push it through all adapters
            episodeFinished = self.environment.episodeFinished()
            for adapter in self.adapters_:
                episodeFinished = adapter.applyEpisodeFinished(episodeFinished)
        
            # first interaction is with the chosen action
            self.interact(forceAction=np.array([action]))

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


    def interact(self, forceAction=None):
        """ run one interaction between agent and environment. The state from
            the environment gets passed through all registered adapters before
            it is presented to the agent. The agent returns an action, which
            goes again through all adapters (in reverse order) and is then 
            executed in the environment. A reward is then passed through the
            adapters to the agent.
            If forceAction is not None, use the given action instead of the
            agent's choice.
        """
        if not self.setupComplete_:
            self._performSetup()
            
        state = self.environment.getState()
        for adapter in self.adapters_:
            state = adapter.applyState(state)
        self.agent.integrateState(state)
        
        if forceAction:
            action = forceAction
        else:
            action = self.agent.getAction()

        for adapter in reversed(self.adapters_):
            action = adapter.applyAction(action)
        self.environment.performAction(action)
        
        reward = self.environment.getReward()        
        for adapter in self.adapters_:
            reward = adapter.applyReward(reward)
        self.agent.giveReward(reward)

