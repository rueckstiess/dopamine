from dopamine.agents.agent import Agent, AgentException
from dopamine.agents.valuebased.faestimator import FAEstimator
from dopamine.fapprox import RBF
from dopamine.tools import History

from numpy import mean, array, r_, c_, atleast_2d, random, equal, ones
import time

class BASAgent(Agent):
    
    alpha = 1.0
    gamma = 0.9
    
    def __init__(self, faClass=RBF):
        """ initialize the agent with the estimatorClass. """
        Agent.__init__(self)
        
        self.amin = -1.
        self.amax = 1.
        self.nres = 3
        
        # store (decision,action) tuples for one action in the list
        self.decisions = []
        
        self.faClass = faClass
    
    def _setup(self, conditions):
        """ if agent is discrete in states and actions create Q-Table. """
        Agent._setup(self, conditions)
        if not (self.conditions['discreteStates'] == False and self.conditions['discreteActions'] == False):
            raise AgentException('BASAgent expects continuous states and actions. Use adapter or a different environment.')
            
        self.estimator = FAEstimator(self.conditions['stateDim'] + self.conditions['actionDim'], 2**self.conditions['actionDim'], self.faClass)
        
        # change history to store bas-extended experiences
        self.history = History(conditions['stateDim']+self.conditions['actionDim'] , 1)
    
    
    def giveReward(self, reward):
        """ override function to store the internal actions in the history. """
        if self.progressCnt == 2:
            self.reward = reward
            self.progressCnt = 0
            if self.loggingEnabled:
                # go through internal decisions and transform them to states, actions, rewards
                olda = array([(self.amax + self.amin) / 2.]*self.conditions['actionDim'])
                for i, (d,a) in enumerate(self.decisions):
                    state = r_[self.state, olda]
                    action = d
                    
                    if i < self.nres-1:
                        reward = 0.
                    else:
                        reward = self.reward
                    
                    self.history.append(state, action, reward)
                    olda = a                   
                    

        else:
            raise AgentException('reward was given before action was returned.')
    
    def _internalDecisions(self, state):
        """ takes a state and queries the estimator several times as a binary search.
            generates (binary) decision and action at each timestep. """
        
        self.decisions = []
        
        a = array([(self.amax + self.amin) / 2.]*self.conditions['actionDim'])
        delta = (self.amax - self.amin) * float(2**(self.nres-1)) / (2**self.nres -1)
        for i in range(self.nres):
            delta = delta/2.
            decision = self.estimator.getBestAction(r_[self.state, a])
            
            # internal epsilon-greedy exploration
            if random.random() < 0.1:
                decision = array([random.randint(2**self.conditions['actionDim'])])

            # turn into binary list
            blist = -1.*ones(self.conditions['actionDim'])
            for i,bit in enumerate(reversed(bin(decision)[2:])):
                if bit == '1':
                    blist[-i-1] = 1.
            
            # update action
            a = a + delta*blist
            self.decisions.append((decision, a))
            
        return a
                
    def _calculate(self):
        """ Return the action with the maximal value for the given state. """
        self.action = self._internalDecisions(self.state)


    def learn(self):
        """ go through whole episode and make Q-value updates. """  
        for i in range(1):
            
            self.estimator.reset()

            for episode in self.history:
                for state, action, reward, nextstate in episode:
                    # # don't consider last state
                    # if equal(state, nextstate).all():
                    #     break

                    qvalue = self.estimator.getValue(state, action)
                    bestnext = self.estimator.getValue(nextstate, self.estimator.getBestAction(nextstate))
                    target = (1-self.alpha) * qvalue + self.alpha * (reward + self.gamma * bestnext)

                    self.estimator.updateValue(state, action, target)

            self.estimator.train()