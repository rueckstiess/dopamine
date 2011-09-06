from fqi import FQIAgent
from faestimator import OrderedFAEstimator
from operator import itemgetter
from random import shuffle

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

    def learn(self):
        """ go through whole episode and make Q-value updates. """  

        for i in range(self.iterations):
            dataset = []
            
            for episode in self.history:
                self.estimator.resetMemory()
                for state, action, reward, nextstate in episode:                    
                    qvalue = self.estimator.getValue(state, action)
                    self.estimator.rememberAction(action)
                    if nextstate != None:
                        bestnext = self.estimator.getValue(nextstate, self.estimator.getBestAction(nextstate))
                    else:
                        bestnext = 0.
                    target = (1-self.alpha) * qvalue + self.alpha * (reward + self.gamma * bestnext)

                    dataset.append([state, action, target])

            if len(dataset) == 0:
                continue
                
            # ground targets to 0 to avoid drifting values
            mintarget = min(map(itemgetter(2), dataset))
            
            # reset estimator (not resetting might lead to faster convergence!)
            if self.resetFA:
                self.estimator.reset()
            for i in range(self.presentations):
                shuffle(dataset)
                for state, action, target in dataset:
                    self.estimator.updateValue(state, action, target-mintarget)
            self.estimator.train()
