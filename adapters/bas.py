from dopamine.adapters import Adapter
import numpy as np

class BinaryActionSearchAdapter(Adapter):
    """ This adapter turns discrete-action environments into
        continuous action environments with the Binary Action
        Search algorithm.
    """
    
    # can be applied to all conditions
    inConditions = {}
    
    # define the conditions of the environment
    outConditions = {}    
    
    def __init__(self, amin, amax, resolution=5):
        # in conditions
        self.inConditions['discreteActions'] = False
        self.inConditions['discreteStates'] = False
        
        # out conditions
        self.outConditions['discreteActions'] = True
        self.outConditions['actionNum'] = 2
        self.outConditions['actionDim'] = 1
        
        self.amax = amax
        self.amin = amin
        self.resolution = resolution
        
    def setExperiment(self, experiment):
        """ give adapter access to the experiment. """
        self.experiment = experiment
        self.outConditions['stateDim'] = self.experiment.conditions['stateDim'] + 1 + 1
    
    def applyState(self, state):
        """ apply transformations to state and return it. """
        self.state = state
        return np.r_[state, self.currAction, self.history]
        
    def applyAction(self, action):
        """ apply transformations to action and return it. """
        decision = int(action.item()) * 2 - 1
        self.history[0] += 1
        self.delta /= 2.
        self.currAction += self.delta * decision
        
        for i in range(self.resolution-1):
            # give reward
            self.experiment.agent.giveReward(0.)
            
            # present next (internal) state
            state = np.r_[self.state, self.currAction, self.history]
            for adapter in self.adapters:
                state = adapter.applyState(state)
            self.experiment.agent.integrateState(state)
            
            # get next (internal) action, including exploration
            action = self.experiment.agent.getAction()
            for adapter in reversed(self.adapters):
                action = adapter.applyAction(action)
            
            decision = int(action.item()) * 2 - 1
            self.history[0] += 1
            self.delta /= 2.
            self.currAction += self.delta * decision
        
        return self.currAction
    
    def applyReward(self, reward):
        """ apply transformations to reward and return it. """
        self.reset()
        return reward
    
    def reset(self):
        """ resets the counter. """
        # get all adapters that are between BAS and the agent (possibly discrete explorers)
        adapters = self.experiment.adapters
        self.adapters = adapters[adapters.index(self)+1:]
        
        self.delta = (self.amax-self.amin) * 2**(self.resolution-1) / (2**self.resolution -1)
        self.currAction = (self.amax+self.amin) / 2.
        self.history = np.zeros(1)
        
