from rllib.environments.maze.maze import Maze
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
    
    def __init__(self):
        Maze.__init__(self, self.matrix, self.goal)
        
    def getReward(self):
        """ compute and return the current reward (i.e. corresponding to the last action performed) """
        Maze.getReward(self)
        
        if self.goal == self.perseus:
            self.reset()
            reward = 1.
        else: 
            reward = 0.
        return reward

    def performAction(self, action):
        """ The action vector is stripped and the only element is cast to integer and given 
            to the super class. 
        """
        Maze.performAction(self, int(action[0]))


    def getState(self):
        """ The agent receives its position in the maze, to make this a fully observable
            MDP problem.
        """
        Maze.getState(self)
        
        obs = array([self.perseus[0] * self.mazeTable.shape[0] + self.perseus[1]])   
        return obs   
