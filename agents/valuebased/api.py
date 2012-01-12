from dopamine.agents.agent import Agent, AgentException
from dopamine.agents.valuebased import FQIAgent
from dopamine.agents.valuebased.vblockestimator import FAEstimator, VectorBlockEstimator
from dopamine.fapprox import RBF, Linear

from numpy import mean, array, r_, c_, atleast_2d, random, equal
from operator import itemgetter
import time
from random import shuffle

class APIAgent(FQIAgent):
    
    def __init__(self, faClass=Linear, resetFA=False, ordered=False, vectorblock=True):
        """ initialize the agent with the estimatorClass. """
        FQIAgent.__init__(self, faClass, resetFA, ordered, vectorblock)
            
    def learn(self):
        """ go through whole episode and make Q-value updates. """  

        for i in range(self.iterations):
            dataset = []
            
            for episode in self.history:
                ret = 0.
                for state, action, reward, nextstate in episode.reversedSamples():
                    qvalue = self.estimator.getValue(state, action)
                    ret += reward
                    target = (1-self.alpha) * qvalue + self.alpha * ret
                    dataset.append([state, action, target])
                    ret *= self.gamma

            if len(dataset) != 0:
                # ground targets to 0 to avoid drifting values
                mintarget = min(map(itemgetter(2), dataset))
                if self.resetFA:
                    self.estimator.reset()
                for i in range(self.presentations):
                    shuffle(dataset)
                    for state, action, target in dataset:
                        self.estimator.updateValue(state, action, target-mintarget)
                self.estimator.train()

