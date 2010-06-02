from dopamine.environments.maze.maze import Maze
from numpy import array


class MDPMaze(Maze):
    
    matrix = array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 1, 0, 0, 0, 0, 1],
                    [1, 0, 0, 1, 0, 0, 1, 0, 1],
                    [1, 0, 0, 1, 0, 0, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 1, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1]])
    
    goal = (7, 7)
    

    # define the conditions of the environment
    conditions = {
      'discreteStates':True,
      'stateDim':1,
      'stateNum':81,
      'discreteActions':True,
      'actionDim':1,
      'actionNum':4, 
      'episodic':True
    }
    
    def __init__(self):
        Maze.__init__(self, self.matrix, self.goal)
    
    
    def reset(self):
        Maze.reset(self)
        self.state = array([self.perseus[0] * self.mazeTable.shape[0] + self.perseus[1]])
    
    def _update(self):
        Maze._update(self)
                
        # set reward       
        if self.goal == self.perseus:
            self.reward = 1.
            # self.reset()
        else:
            self.reward = 0
    
        # set state
        self.state = array([self.perseus[0] * self.mazeTable.shape[0] + self.perseus[1]])
        
