from dopamine.adapters.explorers.explorer import DecayExplorer
from numpy import random, array

class EpsilonGreedyExplorer(DecayExplorer):
    
    # define the conditions of the environment
    inConditions = {'discreteActions':True}    
    
    # define the conditions of the environment
    outConditions = {}
    
    def __init__(self, epsilon, episodeCount=None, actionCount=None):
        """ set the probability epsilon, with which a random action is chosen. """
        DecayExplorer.__init__(self, epsilon, episodeCount, actionCount)
        
        
    def _explore(self, action):
        """ draw random number r uniformly in [0,1]. if r < epsilon, make random move,
            otherwise return action as is.
        """
        if random.random() < self.epsilon:
            action = array([random.randint(self.experiment.conditions['actionNum'])])
                    
        return action
    