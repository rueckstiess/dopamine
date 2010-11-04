from dopamine.agents.direct.direct import DirectAgent
from numpy import *
from numpy.linalg import pinv
from copy import copy

class ReinforceAgent(DirectAgent):
 
    alpha = 10e-3
    sigmadecay = 0.999
 
    def __init__(self):
        DirectAgent.__init__(self)
        
    def _setup(self, conditions):
        """ setup the agent and copy the original parameters. """
        DirectAgent._setup(self, conditions)

    def newEpisode(self):
        """ modify the original controller parameters and store the deltas """
        DirectAgent.newEpisode(self)
                     
    def learn(self):
        # create the derivative of the log likelihoods for each timestep in each episode
        # loglh has the shape of lists (episodes) of lists (timesteps) of arrays of dimensions (s, a)
        loglh = [[dot(s.reshape(self.conditions['stateDim'], 1), 
                     array((a - self.controller.activate(s)) / self.explorer.sigma**2).reshape(1, self.conditions['actionDim'])) for s, a, r, ns in episode] for episode in self.history]
        
        baseline = mean([sum(loglh[ie], axis=0)**2 * sum(e.rewards) for ie, e in enumerate(self.history)], axis=0) / mean([sum(loglh[ie], axis=0)**2 for ie, e in enumerate(self.history)], axis=0)
        gradient = mean([sum(loglh[ie], axis=0) * (sum(e.rewards) - baseline) for ie, e in enumerate(self.history)], axis=0)
        # gradient /= max(abs(gradient))
        # update matrix of linear controller
        self.controller.matrix += self.alpha * gradient
        
        # decay exploration variance
        self.explorer.sigma *= self.sigmadecay
        
        