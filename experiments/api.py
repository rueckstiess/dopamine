from dopamine.experiments.experiment import Experiment, ExperimentException
import numpy as np
import types

class APIExperiment(Experiment):

    def runEpisode(self, reset=True):
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
        if reset:
            for adapter in self.adapters_:
                adapter.reset()
            self.environment.reset()
        
        # interate to a random state in the environment
        while True:
            randomState = self.environment.randomStateReached()
            if randomState:
                break
            self.interact()
        
        # store and remove the current episode from the agent
        randomEpisode = self.agent.history.pop(nonempty=False)
        
        # for each possible action, run one rollout
        for action in range(self.conditions['actionNum']):
            
            # put environment in previous state
            self.environment.resetToState(randomState)
            
            
            #### TODO: is this working correctly???
            ####
            ####
            
            # add stored episode to agent and rebuild the memory
            # self.agent.history.appendEpisode(randomEpisode)
            # self.agent.history.editLastEpisode()
            self.agent.buildMemoryFromEpisode(randomEpisode)
            # print len(self.agent.history)
            
            ####
            ####
            ######################################
            
            
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
            self.numEpisodes += 1
    
    
    def evaluateEpisodes(self, count, reset=True, exploration=False, visualize=True):
        """ executes the original Experiment.runEpisode() function during evaluation. """
        self.funcRunEpisode = self.runEpisode
        self.runEpisode = types.MethodType(Experiment.runEpisode, self)
        valdata = Experiment.evaluateEpisodes(self, count, reset, exploration, visualize)
        self.runEpisode = self.funcRunEpisode
        return valdata

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
        
        action = self.agent.getAction()
        # overwrite action with the forced one (getAction still needs to be called)
        if forceAction:
            action = forceAction

        for adapter in reversed(self.adapters_):
            action = adapter.applyAction(action)
        self.environment.performAction(action)
        
        reward = self.environment.getReward()        
        for adapter in self.adapters_:
            reward = adapter.applyReward(reward)
        self.agent.giveReward(reward)

