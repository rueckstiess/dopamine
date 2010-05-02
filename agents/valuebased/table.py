from numpy import ones, ndarray
import types

from rllib.agents.valuebased.estimator import Estimator

class TableException(Exception):
    pass
    
class Table(Estimator):
    
    discreteStates = True
    discreteActions = True
    
    def __init__(self, stateNum, actionNum):
        """ initialize with the number of states and actions. the table
            values are all set to zero.
        """
        self.stateNum = stateNum
        self.actionNum = actionNum
        
        self.initialize(0.0)
 
    def getMaxAction(self, state):
        """ expects a scalar or a list or array with one element. """
        state = self._forceScalar(state)
        return max(self.values[state, :])
        
    def getValue(self, state, action):
        """ returns the value of the given (state,action) pair. """
        state = self._forceScalar(state)
        action = self._forceScalar(action)
        return self.values[state, action]
        
    def updateValue(self, state, action, value):
        """ sets a new value of the given (state, action) pair. """
        state = self._forceScalar(state)
        action = self._forceScalar(action)
        self.values[state, action] = value

    def initialize(self, value=0.0):
        """ Initialize the whole table with the given value. """
        self.values = ones((self.stateNum, self.actionNum)) * value
    
    def _forceScalar(self, value):
        """ accepts scalars, lists and arrays. scalars are just passed through, while
            lists and arrays must have one single element. that element is passed back
            as a scalar. lists/arrays with more than one element raise an Exception.
        """
        if type(value) in [ndarray, types.ListType]:
            if len(value) > 1:
                raise TableException('Table accepts only scalars or lists/arrays with one element as state or action.')
            if type(value) == ndarray:
                value = value.item()
            else:
                value = value[0]

        return int(value)
        
           

    