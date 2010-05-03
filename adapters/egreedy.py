from rllib.adapters import Adapter
from numpy import random, array

class EpsilonGreedyExplorer(Adapter):
    
    # define the conditions of the environment
    inConditions = {'discreteStates':True, 'discreteActions':True}    
    
    # define the conditions of the environment
    outConditions = {}
    
    def __init__(self, epsilon, decay=0.999):
        """ set the probability epsilon, with which a random action is chosen. """
        self.epsilon = epsilon
        self.decay = decay
        self.agent = None
        
    def applyAction(self, action):
        """ apply transformations to action and return it. """
        if random.random() < self.epsilon:
            action = array([random.randint(self.agent.conditions['actionNum'])])
        
        self.epsilon *= self.decay
        return action
    