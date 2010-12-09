from dopamine.agents.direct.direct import DirectAgent
from dopamine.agents.direct.controllers.linear import LinearController
from numpy import *
from numpy.linalg import pinv
from copy import copy

class ReinforceAgent(DirectAgent):
 
    alpha = 10e-2
    alpha_sigma = 10e-3
    sigmadecay = 0.005
                          
    def learn(self):
        # create the derivative of the log likelihoods for each timestep in each episode
        # loglh has the shape of lists (episodes) of lists (timesteps) of arrays of dimensions (s, a)
        
        loglh_mu = []
        for episode in self.history:
            inner_loglh_mu = []
            for s, a, r, ns in episode:
                # calculate dpi/dmu
                rl_error = array((a - self.controller.predict(s))).reshape(1, self.conditions['actionDim'])
                # calculate dpi/dtheta
                inner_loglh_mu.append(self.controller.dOutdTheta(s, rl_error))
                # print inner_loglh_mu[-1]
            loglh_mu.append(inner_loglh_mu)
        
        baseline = mean([sum(loglh_mu[ie], axis=0)**2 * mean(e.rewards) for ie, e in enumerate(self.history)], axis=0) / mean([sum(loglh_mu[ie], axis=0)**2 for ie, e in enumerate(self.history)], axis=0)
        gradient = mean([sum(loglh_mu[ie], axis=0) * ((mean(e.rewards) - baseline)) for ie, e in enumerate(self.history)], axis=0)
        
        # update parameters of controller
        self.controller.parameters = self.controller.parameters + self.alpha * gradient.flatten()
        print gradient
        
        for explorer in self.experiment.explorers:
            if hasattr(explorer, 'sigmaAdaptation') and explorer.sigmaAdaptation:
                loglh_sigma = []
                for episode in self.history:
                    inner_loglh_sigma = []
                    for s, a, r, ns in episode:
                        # calculate dpi/dmu
                        rl_error = array((a - self.controller.predict(s))).reshape(1, self.conditions['actionDim'])
                        # calculate dpi/dtheta
                        inner_loglh_sigma.append(explorer.getDerivative(s, rl_error))
                    loglh_sigma.append(inner_loglh_sigma)
        
                baseline = mean([sum(loglh_sigma[ie], axis=0)**2 * mean(e.rewards) for ie, e in enumerate(self.history)], axis=0) / mean([sum(loglh_sigma[ie], axis=0)**2 for ie, e in enumerate(self.history)], axis=0)
                gradient = mean([sum(loglh_sigma[ie], axis=0) * ((mean(e.rewards) - baseline)) for ie, e in enumerate(self.history)], axis=0)
            
                # update parameters of explorer
                explorer.sigma = explorer.sigma + self.alpha_sigma * gradient.flatten()
            else:
                # decay exploration variance
                for explorer in self.experiment.explorers:
                    explorer.sigma -= self.sigmadecay
        
        