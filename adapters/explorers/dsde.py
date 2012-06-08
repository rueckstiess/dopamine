from dopamine.adapters.explorers.explorer import DecayExplorer
import numpy as np
from copy import copy

class DiscreteSDExplorer(DecayExplorer):
    
    # define the conditions of the environment
    inConditions = {'discreteStates':True, 'discreteActions':True}
            

    def __init__(self, epsilon, episodeCount=None, actionCount=None):
        DecayExplorer.__init__(self, epsilon, episodeCount, actionCount)

        # active is now property, this is the helper variable
        self.active_ = True   
        self.oldTable = None

    def reset(self):
        self._permuteTableRows()  


    def _permuteTableRows(self):
        if self.experiment == None:
            return

        table = self.experiment.agent.estimator.values

        if self.oldTable == None:
            self.oldTable = copy(table)

        for i, r in enumerate(table):
            if np.random.random() < self.epsilon:
                table[i] = np.random.permutation(r)


    def applyEpisodeFinished(self, episodeFinished):
        """ apply transformations to episodeFinished and return it. """
        episodeFinished = DecayExplorer.applyEpisodeFinished(self, episodeFinished)
        
        if episodeFinished:
            if self.oldTable != None:
                self.experiment.agent.estimator.values[:] = self.oldTable
                self.oldTable = None

        return episodeFinished

    
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

