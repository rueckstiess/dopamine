from dopamine.agents.direct.direct import DirectAgent
import numpy as np
from copy import copy

class ENACAgent(DirectAgent):
 
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
                rl_error = np.array((a - self.controller.predict(s))).reshape(1, self.conditions['actionDim'])
                # calculate dpi/dtheta
                inner_loglh_mu.append(self.controller.dOutdTheta(s, rl_error))
            loglh_mu.append(inner_loglh_mu)
        
        X = np.zeros((0, len(self.controller.parameters)+1))
        for ie, e in enumerate(self.history):
            row = np.atleast_2d(np.r_[np.sum(loglh_mu[ie], axis=0).flatten(), np.array([1])])
            X = np.r_[X, row]

        Y = np.array([[sum(e.rewards) for e in self.history]]).T
        
        gradient = np.dot(np.linalg.pinv(X), Y)[:len(self.controller.parameters)]

        # update parameters of controller
        self.controller.parameters = self.controller.parameters + self.alpha * gradient.flatten()
        
        # decay exploration variance
        for explorer in self.experiment.explorers:
            explorer.sigma -= self.sigmadecay

