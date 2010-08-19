from dopamine.adapters import Adapter
import random as pyrnd
from numpy import *
from numpy.linalg import norm
from matplotlib import pyplot as plt

class VQActionDiscretizationAdapter(Adapter):
    """ This adapter discretizes the action space by means of a k-mean vector 
        quantization. numActionVectors gives the number of vectors used to 
        discretize. A value of 0 disables discretization.
    """
      
    requirePretraining = 100
      
    def __init__(self, numActionVectors):
        self.numActionVectors = numActionVectors
        
        if numActionVectors > 0:
            self.outConditions['discreteActions'] = True
            self.outConditions['actionDim'] = 1
            self.outConditions['actionNum'] = numActionVectors
                    
        self.originalActionDim = None        
        self.originalActions = []  
        self.initialized = False       
        self.alpha = 0.1
        
        
    def setExperiment(self, experiment):
        """ give adapter access to the experiment. """
        self.experiment = experiment


    def _findClosestCluster(self, clusters, vec):
        """ returns the clostes cluster center to vec. """
        dist = clusters - vec
        diff = [norm(r, 2) for r in dist]
        return argmin(diff)
    

    def applyAction(self, action):
        if self.numActionVectors <= 0:
            return action

        self.originalActions.append(action)
        
        # initialize for first request
        if not self.originalActionDim:
            # get action dimension from action passed in
            self.originalActionDim = len(action.flatten())
            # create random vectors
            self.actionVectors = random.random((self.numActionVectors, self.originalActionDim))
        
    
        if not self.initialized and len(self.originalActions) >= self.requirePretraining:
            self.sampleClusters()
            self.adaptClusters()
            self.initialized = True
        
        action = self._findClosestCluster(self.actionVectors, action.reshape(1, self.originalActionDim))        
        return array([action])

    
    def sampleClusters(self):
        # sample for initialization
        for i,sv in enumerate(pyrnd.sample(self.originalActions, self.numActionVectors)):
            self.actionVectors[i,:] = sv

        
    def adaptClusters(self):
        """ adapts the cluster centers to better represent the data density. """
        alpha = self.alpha
          
        for i in range(10):
            for a in self.originalActions:
                winner = self._findClosestCluster(self.actionVectors, a)
                self.actionVectors[winner, :] -= alpha * (self.actionVectors[winner, :] - a)
            alpha *= 0.9

        # delete saved states and actions
        self.originalActions = []
