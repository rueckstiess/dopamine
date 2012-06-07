from dopamine.adapters.explorers import Explorer
import numpy as np
from copy import copy

class DiscreteSDExplorer(Explorer):
    
    # define the conditions of the environment
    inConditions = {'discreteStates':True, 'discreteActions':True}
            

    def __init__(self, epsilon=0.5, decay=0.9999):        
        # active is now property, this is the helper variable
        self.active_ = True   

        # probability and its decay to permute a row in the value table
        self.epsilon = epsilon
        self.decay = decay

        self.oldTable = None

    def reset(self):
        self._permuteTableRows()  


    def _permuteTableRows(self):
        table = self.experiment.agent.estimator.values

        if self.oldTable == None:
            self.oldTable = copy(table)

        for i, r in enumerate(table):
            if np.random.random() < self.epsilon:
                table[i] = np.random.permutation(r)


    def applyEpisodeFinished(self, episodeFinished):
        """ apply transformations to episodeFinished and return it. """
        if episodeFinished:
            if self.oldTable != None:
                self.experiment.agent.estimator.values[:] = self.oldTable
                self.oldTable = None

        return episodeFinished

    
    def _explore(self, action):
        # disable exploration if tau drops below 0.01
        if self.epsilon < 0.0001:
            self.active = False
        
        if self.active:
            self.epsilon *= self.decay

        return np.array([action])

    def _setActive(self, active):
        if active:
            self._permuteTableRows()
        else:
            if self.oldTable != None:
                self.experiment.agent.estimator.values[:] = self.oldTable
                self.oldTable = None

        self.active_ = active

    def _getActive(self):
        return self.active_

    active = property(_getActive, _setActive)

