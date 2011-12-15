from dopamine.agents.agent import Agent, AgentException
from dopamine.agents.valuebased.faestimator import FAEstimator
from dopamine.fapprox import RBF

from numpy import mean, array, r_, c_, atleast_2d, random, equal
from operator import itemgetter
from random import shuffle
import time

class FQIAgent(Agent):
    
    alpha = 1.0
    gamma = 0.9
    iterations = 1
    presentations = 1
    
    def __init__(self, faClass=RBF, resetFA=True, ordered=False):
        """ initialize the agent with the estimatorClass. """
        Agent.__init__(self)
        self.faClass = faClass
        self.resetFA = resetFA
        self.ordered = ordered

    
    def _setup(self, conditions):
        """ if agent is discrete in states and actions create Q-Table. """
        Agent._setup(self, conditions)
        if not (self.conditions['discreteStates'] == False and self.conditions['discreteActions']):
            raise AgentException('FQIAgent expects continuous states and discrete actions. Use adapter or a different environment.')
        
        self.estimator = FAEstimator(self.conditions['stateDim'], self.conditions['actionNum'], faClass=self.faClass, ordered=self.ordered)
    

    def _calculate(self):
        self.action = self.estimator.getBestAction(self.state)
    
    
    def newEpisode(self):
        """ reset the memory. """
        Agent.newEpisode(self)

        if self.ordered:
            self.estimator.resetMemory()
    

    def giveReward(self, reward):
        """ additionally remember the chosen action to not draw it again. """
        if self.ordered:
            self.estimator.rememberAction(self.action)
        
        Agent.giveReward(self, reward)    

    
    def learn(self):
        """ go through whole episode and make Q-value updates. """  

        for i in range(self.iterations):
            dataset = []
            
            for episode in self.history:
                if self.ordered:
                    self.estimator.resetMemory()

                for state, action, reward, nextstate in episode:                    
                    qvalue = self.estimator.getValue(state, action)
                    if self.ordered:
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

