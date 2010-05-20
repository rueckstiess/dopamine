from dopamine.adapters import Adapter
from numpy import *

class VQDiscretizationAdapter(Adapter):
    """ This adapter discretizes the state and action space by means
        of a k-mean vector quantization. numStateVectors and numActionVectors
        gives the number of vectors used to discretize. A value of 0
        disables discretization.
    """
        
    def __init__(self, numStateVectors, numActionVectors):
        self.numStateVectors = numStateVectors
        self.numActionVectors = numActionVectors
        
        if numStateVectors > 0:
            self.outConditions['stateDim'] = 1
            self.outConditions['stateNum'] = numStateVectors
        
        if numActionVectors > 0:
            self.outConditions['actionDim'] = 1
            self.outConditions['actionNum'] = numActionVectors
            
        self.originalStateDim = None
        self.originalActionDim = None
        
        self.alpha = 0.3
        
        
    def setExperiment(self, experiment):
        """ give adapter access to the experiment. """
        self.experiment = experiment

    def applyState(self, state):
        if self.numStateVectors <= 0:
            return state
            
        # initialize if necessary
        if not self.originalStateDim:
            self.originalStateDim = len(state.flatten())
            self.stateVectors = random.random((self.numStateVectors, self.originalStateDim))
        
        # calculate distances to each vector
        dist = self.stateVectors - state.reshape(1, self.originalStateDim)
        diff = [sum(map(lambda x: x**2, r)) for r in dist]
        # find winner
        state = argmin(diff)
        # move winner closer to state
        self.stateVectors[state, :] -= self.alpha * dist[state,:]
        
        return array([state])

    def applyAction(self, action):
        if self.numActionVectors <= 0:
            return action
        
        # initialize if necessary
        if not self.originalActionDim:
            self.originalActionDim = len(action.flatten())
            self.actionVectors = random.random((self.numActionVectors, self.originalActionDim))
        
        # calculate distances to each vector
        dist = self.actionVectors - action.reshape(1, self.originalActionDim)
        diff = [sum(map(lambda x: x**2, r)) for r in dist]
        # find winner
        action = argmin(diff)
        # move winner closer to state
        self.actionVectors[action, :] -= self.alpha * dist[action,:]
        
        return array([action])