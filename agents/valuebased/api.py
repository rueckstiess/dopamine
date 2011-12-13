from dopamine.agents.agent import Agent, AgentException
from dopamine.agents.valuebased.vblockestimator import VectorBlockEstimator
from dopamine.fapprox import RBF

from numpy import mean, array, r_, c_, atleast_2d, random, equal
from operator import itemgetter
import time
from random import shuffle

class APIAgent(Agent):
    
    alpha = 1.0
    gamma = 0.9
    presentations = 1
    iterations = 1
    
    def __init__(self, resetFA=True):
        """ initialize the agent with the estimatorClass. """
        Agent.__init__(self)
        self.resetFA = resetFA
    
    def _setup(self, conditions):
        """ if agent is discrete in states and actions create Q-Table. """
        Agent._setup(self, conditions)
        if not (self.conditions['discreteStates'] == False and self.conditions['discreteActions']):
            raise AgentException('FQIAgent expects continuous states and discrete actions. Use adapter or a different environment.')
            
        self.estimator = VectorBlockEstimator(self.conditions['stateDim'], self.conditions['actionNum'])
    
    def _calculate(self):
        self.action = self.estimator.getBestAction(self.state)
    
    def learn(self):
        """ go through whole episode and make Q-value updates. """  

        for i in range(self.iterations):
            dataset = []
            
            for episode in self.history:
                for e, (state, action, reward, nextstate) in enumerate(episode):
                    qvalue = self.estimator.getValue(state, action)
                    target = (1-self.alpha) * qvalue + self.alpha * sum([self.gamma**k * r for k,r in enumerate(episode.rewards[e:])])
                    dataset.append([state, action, target])

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

