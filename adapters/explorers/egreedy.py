from dopamine.adapters.explorers import Explorer
from numpy import random, array

class EpsilonGreedyExplorer(Explorer):
    
    # define the conditions of the environment
    inConditions = {'discreteActions':True}    
    
    # define the conditions of the environment
    outConditions = {}
    
    def __init__(self, epsilon, decay=0.999):
        """ set the probability epsilon, with which a random action is chosen. """
        Explorer.__init__(self)
        
        self.epsilon = epsilon
        self.decay = decay
        
        
    def _explore(self, action):
        """ draw random number r uniformly in [0,1]. if r < epsilon, make random move,
            otherwise return action as is.
        """
        
        # disable exploration if tau drops below 0.01
        if self.epsilon < 0.0001:
            self.active = False
            return array([action])
        
        if random.random() < self.epsilon:
            action = array([random.randint(self.experiment.conditions['actionNum'])])
        
        if self.active:
            self.epsilon *= self.decay
            
        return action
    