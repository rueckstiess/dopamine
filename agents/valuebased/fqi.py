from dopamine.agents import Agent
from dopamine.agents.agent import AgentException
from dopamine.agents.valuebased import *
from dopamine.tools.utilities import one_to_n

from numpy import mean, array, r_, c_, atleast_2d, random, equal
from operator import itemgetter
import time

class FQIAgent(Agent):
    
    alpha = 1.0
    gamma = 0.9
    iterations = 1
    
    def __init__(self, estimatorClass=NNEstimator):
        """ initialize the agent with the estimatorClass. """
        Agent.__init__(self)
        self.estimatorClass = estimatorClass
    
    def _setup(self, conditions):
        """ if agent is discrete in states and actions create Q-Table. """
        Agent._setup(self, conditions)
        if not (self.conditions['discreteStates'] == False and self.conditions['discreteActions']):
            raise AgentException('FQIAgent expects continuous states and discrete actions. Use adapter or a different environment.')
            
        self.estimator = self.estimatorClass(self.conditions['stateDim'], self.conditions['actionNum'])
    
    def _calculate(self):
        self.action = self.estimator.getBestAction(self.state)
    
    def learn(self):
        """ go through whole episode and make Q-value updates. """  

        for i in range(self.iterations):
            dataset = []
            
            for episode in self.history:
                for state, action, reward, nextstate in episode:
                    # don't consider last state
                    # if equal(state, nextstate).all():
                    #     break
                    
                    qvalue = self.estimator.getValue(state, action)
                    bestnext = self.estimator.getValue(nextstate, self.estimator.getBestAction(nextstate))
                    target = (1-self.alpha) * qvalue + self.alpha * (reward + self.gamma * bestnext)

                    dataset.append([state, action, target])

            if len(dataset) == 0:
                continue
                
            # ground targets to 0 to avoid drifting values
            mintarget = min(map(itemgetter(2), dataset))
            self.estimator.reset()
            for state, action, target in dataset:
                self.estimator.updateValue(state, action, target-mintarget)
            self.estimator.train()

