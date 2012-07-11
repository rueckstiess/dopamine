from dopamine.adapters.explorers.explorer import DecayExplorer
from dopamine.fapprox import Linear
import numpy as np


class EpsilonGreedySDExplorer(DecayExplorer):
    
    # define the conditions of the environment
    inConditions = {'discreteStates': False, 'discreteActions':True}    
    
    # define the conditions of the environment
    outConditions = {}
    
         
    def setExperiment(self, experiment):
        DecayExplorer.setExperiment(self, experiment)

        # create random exploration mapping
        self.explMapping = Linear(self.experiment.conditions['stateDim'], self.experiment.conditions['actionNum'])
        self.randomizeMapping()


    def applyState(self, state):
        """ save current state for _explore() method later on. """
        DecayExplorer.applyState(self, state)

        self.state = state
        return state


    def _explore(self, action):
        """ add an episode-specific offset to each action """
        if np.random.random() < self.epsilon:
            exploration = self.explMapping.predict(self.state).flatten()
            action = np.argmax(exploration)

        return action 


    def randomizeMapping(self):
        self.explMapping.parameters = np.random.normal(0., 0.1, size=self.explMapping.parameters.shape)


    def applyEpisodeFinished(self, episodeFinished):
        """ apply transformations to episodeFinished and return it. """
        DecayExplorer.applyEpisodeFinished(self, episodeFinished)

        # at end of episode, randomize the exploration parameters
        if episodeFinished:
            self.randomizeMapping()

        return episodeFinished
    