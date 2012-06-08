from dopamine.adapters.explorers.explorer import DecayExplorer
from numpy import random, array, where, exp

class BoltzmannExplorer(DecayExplorer):
    
    # define the conditions of the environment
    inConditions = {'discreteActions':True}    
    
    # define the conditions of the environment
    outConditions = {}
    
    def __init__(self, epsilon, episodeCount=None, actionCount=None):
        """ set the probability epsilon, with which a random action is chosen. """
        DecayExplorer.__init__(self, epsilon, episodeCount, actionCount)
        
        self.state = None
        
    def applyState(self, state):
        self.state = state
        return state
    
    def _explore(self, action):
        """ draw random move with probability proportional to the action's value.
        """
        pdf = [exp(self.experiment.agent.estimator.getValue(self.state, array([a]))/self.epsilon) \
            for a in range(self.experiment.conditions['actionNum'])]
        pdf /= sum(pdf)
        cdf = [sum(pdf[:i+1]).item() for i in range(len(pdf))]
                    
        r = random.random()   
        action = sum(array(cdf) < r)

        return array([action])