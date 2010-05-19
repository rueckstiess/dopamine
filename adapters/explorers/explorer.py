from dopamine.adapters import Adapter

class Explorer(Adapter):
    
    # define the conditions of the environment
    inConditions = {}    
    
    # define the conditions of the environment
    outConditions = {}
            
    def __init__(self):
        Adapter.__init__(self)
        
        # set this to False to turn off exploration
        self.active = True        
    
    def applyAction(self, action):
        """ apply transformations to action and return it. """
        if self.active:
            action = self._explore(action)
            self.experiment.agent.action = action
        
        return action
    
    def _explore(self, action):
        pass
    
    

