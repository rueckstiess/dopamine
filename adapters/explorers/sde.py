from dopamine.adapters.explorers import Explorer
from numpy import random, array, dot

class StateDependentExplorer(Explorer):
    
    # define the conditions of the environment
    inConditions = {'discreteActions':False}    
    
    # define the conditions of the environment
    outConditions = {}
    
    def __init__(self, sigma=0.2):
        """ set the variance sigma for the gaussian distribution. """
        Explorer.__init__(self)
        self.sigma = sigma
     
    def setExperiment(self, experiment):
        Explorer.setExperiment(self, experiment)
           
        # parameters for the linear exploration function
        self.theta = random.normal(0, self.sigma, \
            (self.experiment.conditions['stateDim'], self.experiment.conditions['actionDim']))
        
    def applyState(self, state):
        """ save current state for _explore() method later on. """
        self.state = state
        return state

    def _explore(self, action):
        """ add an episode-specific offset to each action """
        exploration = self._explFunction(self.state).flatten()
        action += exploration

        return action 

    def _explFunction(self, state):
        return dot(state.reshape(1, self.experiment.conditions['stateDim']), self.theta)
    
    def applyEpisodeFinished(self, episodeFinished):
        """ apply transformations to episodeFinished and return it. """
        if episodeFinished:
            # at end of episode, randomize the exploration parameters
            self.theta = random.normal(0, self.sigma, \
                (self.experiment.conditions['stateDim'], self.experiment.conditions['actionDim']))
        return episodeFinished
    
