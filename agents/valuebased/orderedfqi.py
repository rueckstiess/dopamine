from fqi import FQIAgent
from faestimator import OrderedFAEstimator

class OrderedFQIAgent(FQIAgent):
    """ This agent uses the OrderedFAEstimator class to allow
        actions to be chosen only once during an episode. It tells
        the estimator which actions have been chosen before, and
        resets the estimator's memory on beginning of a new episode.
    """
    
    def _setup(self, conditions):
        """ use OrderedFAEstimator instead of FAEstimator. """
        FQIAgent._setup(self, conditions)
        self.estimator = OrderedFAEstimator(self.conditions['stateDim'], self.conditions['actionNum'], self.faClass)
    
    def newEpisode(self):
        """ reset the memory. """
        FQIAgent.newEpisode(self)
        self.estimator.resetMemory()
    
    def giveReward(self, reward):
        """ additionally remember the chosen action to not draw it again. """
        self.estimator.rememberAction(self.action)
        FQIAgent.giveReward(self, reward)
