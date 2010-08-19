from dopamine.adapters import Adapter
import random as pyrnd
from numpy import *
from numpy.linalg import norm
from matplotlib import pyplot as plt

class VQDiscretizationAdapter(Adapter):
    """ This adapter discretizes the state space by means of a k-mean vector 
        quantization. numStateVectors gives the number of vectors used to 
        discretize. A value of 0 disables discretization.
    """

    requirePretraining = 100
      
    def __init__(self, numStateVectors):
        self.numStateVectors = numStateVectors
        
        if numStateVectors > 0:
            self.outConditions['discreteStates'] = True
            self.outConditions['stateDim'] = 1
            self.outConditions['stateNum'] = numStateVectors
                    
        self.originalStateDim = None        
        self.originalStates = []   
        self.initialized = False     
        self.alpha = 0.1
        
        
    def setExperiment(self, experiment):
        """ give adapter access to the experiment. """
        self.experiment = experiment


    def _findClosestCluster(self, clusters, vec):
        """ returns the clostes cluster center to vec. """
        dist = clusters - vec
        # diff = [sum(map(lambda x: x**2, r)) for r in dist]
        diff = [norm(r, 2) for r in dist]
        return argmin(diff)
    

    def applyState(self, state):
        if self.numStateVectors <= 0:
            return state

        self.originalStates.append(state)
        
        # initialize if necessary
        if not self.originalStateDim:
            self.originalStateDim = len(state.flatten())
            self.stateVectors = random.random((self.numStateVectors, self.originalStateDim))
               
        if not self.initialized and len(self.originalStates) >= self.requirePretraining:
            self.sampleClusters()
            self.adaptClusters()
            self.initialized = True
        
        state = self._findClosestCluster(self.stateVectors, state.reshape(1, self.originalStateDim))      
        
        return array([state])

    
    def sampleClusters(self):
        # sample for initialization
        for i,sv in enumerate(pyrnd.sample(self.originalStates, self.numStateVectors)):
            self.stateVectors[i,:] = sv

        
    def adaptClusters(self):
        """ adapts the cluster centers to better represent the data density. """
        alpha = self.alpha
          
        for i in range(10):
            for s in self.originalStates:
                winner = self._findClosestCluster(self.stateVectors, s)
                self.stateVectors[winner, :] -= alpha * (self.stateVectors[winner, :] - s)
            alpha *= 0.9

        # delete saved states and actions, decay alpha
        self.originalStates = []
