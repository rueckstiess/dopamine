from dopamine.adapters import Adapter
import numpy as np

class Explorer(Adapter):
    
    # define the conditions of the environment
    inConditions = {}    
    
    # define the conditions of the environment
    outConditions = {}
            
    def __init__(self):
        Adapter.__init__(self)
        
        # set this to False to turn off exploration
        self.active = True        
    
    def applyAction(self, action):
        """ apply transformations to action and return it. """
        if self.active:
            action = self._explore(action)
            # tell agent the action that was executed (for the history)
            self.experiment.agent.action = action
        
        return action
    
    def _explore(self, action):
        return action
    

class DecayExplorer(Explorer):

    finalFactor = 0.001

    def __init__(self, epsilon, episodeCount=None, actionCount=None):
        """ DecayExplorer is an explorer base class that has exploration decay, i.e.
            the amount of exploration weakens exponentially over time. epsilon is the
            initial parameter (can mean different things for different explorers), 
            which reduced over time. if episodeCount is given, epsilon reduces to 
            1/1000 of the initial value in the given number of episodes. if actionCount
            is given, epsilon reduces to 1/1000 of the initial value in the given 
            number of actions executed. actionCount takes priority if both values
            are given. In either case, after epsilon is 1/1000 of its initial value,
            exploration automatically deactivates.
        """

        Explorer.__init__(self)

        self.episodeCount = episodeCount
        self.actionCount = actionCount
        self.epsilon = epsilon
        self.initialEpsilon = epsilon

        if self.episodeCount:
            self.decay = np.power(self.finalFactor, 1./self.episodeCount)

        if self.actionCount:
            self.decay = np.power(self.finalFactor, 1./self.actionCount)
            self.episodeCount = None

    def resetExploration(self):
        self.epsilon = self.initialEpsilon

    def applyAction(self, action):
        action = Explorer.applyAction(self, action)

        if self.actionCount and self.active:
            self.epsilon *= self.decay

        if self.epsilon <= self.initialEpsilon * self.finalFactor:
            self.active = False 

        return action

    def applyEpisodeFinished(self, episodeFinished):
        if episodeFinished and self.episodeCount and self.active:
            self.epsilon *= self.decay

        return episodeFinished

