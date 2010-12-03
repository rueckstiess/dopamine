from numpy import ndarray, ones
import types

from dopamine.agents.actorcritic.critics.critic import Critic

class TableException(Exception):
    pass
    
class TableCritic(Critic):
    
    conditions = {'discreteStates':True, 'discreteActions':True}
    
    def __init__(self, stateNum):
        """ initialize with the number of states. the table
            values are all set to zero.
        """
        self.stateNum = stateNum
        self.initialize(0.0)
 
    def getValue(self, state):
        """ returns the value of the given state. """
        state = self._forceScalar(state)
        return self.values[state]
        
    def updateValue(self, state, value):
        """ sets a new value of the given state. """
        state = self._forceScalar(state)
        self.values[state] = value

    def initialize(self, value=0.0):
        """ Initialize the whole table with the given value. """
        self.values = ones(self.stateNum) * value
    
    def _forceScalar(self, value):
        """ accepts scalars, lists and arrays. scalars are just passed through, while
            lists and arrays must have one single element. that element is passed back
            as a scalar. lists/arrays with more than one element raise an Exception.
        """
        if type(value) in [ndarray, types.ListType]:
            if len(value) > 1:
                raise TableException('TableCritic accepts only scalars or lists/arrays with one element as state or action.')
            if type(value) == ndarray:
                value = value.item()
            else:
                value = value[0]

        return int(value)
        
    def __str__(self):
        return str(self.values)

