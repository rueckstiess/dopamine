from dopamine.environments.maze.maze import Maze
from numpy import array, zeros, inf
from random import choice

class TMaze(Maze):   
    """
    #############
    ###########*#
    #.          #
    ########### #
    ############# 
    
    1-in-n encoding for observations.
    """

    # define the conditions of the environment
    conditions = {
      'discreteStates':False,
      'stateDim':4,
      'stateNum':inf,
      'discreteActions':True,
      'actionDim':1,
      'actionNum':4, 
      'episodic':True
    }
    
    finalReward = 4
    bangReward = -0.1
    length = 1
    stochAction = 0.
    
    def __init__(self):
        # initial position always at the left side of the corridor
        self.initPos = [(2, 1)]
        
        # create the maze matrix
        columns = [[1] * 5]
        for dummy in range(self.length):
            columns.append([1, 1, 0, 1, 1])
        columns.append([1, 0, 0, 0, 1])
        columns.append([1] * 5)
        self.matrix = array(columns).T
        
        Maze.__init__(self, self.matrix, self.initPos)
        
    def reset(self):
        """ resets the maze and chooses a random goal (up or down). """
        Maze.reset(self)
        
        self.goUp = choice([True, False])
        self.goUp = True
        if self.goUp:
            self.goal = (3, self.length + 1)
        else:
            self.goal = (1, self.length + 1)
    
    
    def _update(self):
        """ integrate self.action and set new self.state and self.reward.
            self.state is coded as follows:
                index 0: if in start state, this indicates goal up
                index 1: if in start state, this indicates goal down
                index 2: this indicates robot is neither at start state nor at the junction
                index 3: this indicates robot is at the junction.
        """
        Maze._update(self)
        
        # set new state
        self.state = zeros(4)
        if self.perseus == self.initPos[0]:
            if self.goUp:
                self.state[0] = 1
            else:
                self.state[1] = 1
        elif self.perseus[1] == self.length + 1:
            self.state[2] = 1
        else:
            self.state[3] = 1
        
        # set new reward
        if self.perseus[1] == self.length + 1:
            if abs(self.perseus[0] - self.goal[0]) == 2:
                # bad choice taken
                self.perseus = self.goal
                self.reward = self.bangReward

        
    
