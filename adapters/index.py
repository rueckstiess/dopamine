from dopamine.adapters import Adapter
from numpy import array, inf, ones

class IndexingAdapter(Adapter):
    """ This adapter chooses selectively from the state and action vector
        and only passes certain indices of these vectors. Specify the
        indices in the initialization. Pass None, if you don't want the
        state or action vector indexed.
        
        Example:
            IndexingAdapter([1, 3, 5], [0, 1])
            
            This adapter would only pass state indices 1, 3 and 5 to the
            agent, thus making it a 3-dimensional environment. it would also
            only pass action indices 0 and 1 back to the agent. Note: state
            indexing is much more common (e.g. to hide certain information
            from the agent).
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
