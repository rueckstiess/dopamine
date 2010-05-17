from rllib.adapters import Adapter

class Explorer(Adapter):
    
    # define the conditions of the environment
    inConditions = {}    
    
    # define the conditions of the environment
    outConditions = {}
            
    def applyAction(self, action):
        """ apply transformations to action and return it. """
        action = self._explore(action)
        self.experiment.agent.action = action
        return action
    
    def _explore(self, action):
        pass
    
    

