from rllib.adapters import Adapter
from numpy import array, inf, ones

class IndexingAdapter(Adapter):
    """ 
    """
    
    # can be applied to all conditions
    inConditions = {}
    
    # define the conditions of the environment
    outConditions = {}   
    
    def __init__(self, stateIndices, actionIndices):
        self.stateIndices = stateIndices
        self.actionIndices = actionIndices 
    
    def setExperiment(self, experiment):
        Adapter.setExperiment(self, experiment)
        
        if self.stateIndices and not self.experiment.conditions['discreteStates']:
            self.outConditions['stateDim'] = len(self.stateIndices)
        if self.actionIndices and not self.experiment.conditions['discreteActions']:
            self.outConditions['actionDim'] = len(self.actionIndices)        
        
    def applyState(self, state):
        if self.stateIndices and not self.experiment.conditions['discreteStates']:
            state = state.take(self.stateIndices)
        return state
        
    def applyAction(self, action):
        if self.actionIndices and not self.experiment.conditions['discreteActions']:
            action = action.take(self.actionIndices)
        return action          
