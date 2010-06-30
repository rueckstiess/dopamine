from dopamine.environments import MDPMaze
from dopamine.agents import QAgent, SARSAAgent
from dopamine.experiments import Experiment
from dopamine.adapters import MakeEpisodicAdapter, EpsilonGreedyExplorer

from matplotlib import pyplot as plt
from numpy import *
import time


agent = QAgent()
# agent = SARSAAgent()

environment = MDPMaze()
experiment = Experiment(environment, agent)
experiment.addAdapter(MakeEpisodicAdapter(1000))


explorer = EpsilonGreedyExplorer(0.3, 0.9999)
experiment.addAdapter(explorer)

plt.ion()

for i in range(1000):
    # run one episode and learn
    experiment.runEpisode(reset=True)
    agent.learn()

    shape = environment.mazeTable.shape
    
    # plot max_a Q(s, a) for each state s
    plt.clf()
    plt.pcolor(agent.estimator.values.max(axis=1).reshape(shape), cmap='gray') 

    # create meshgrid for quiver 
    X, Y = meshgrid(arange(0., shape[0], 1.0), arange(0., shape[1], 1.0))
    X += 0.5
    Y += 0.5
    
    # calculate values as softmax
    values = agent.estimator.values.copy()
    nozeros = atleast_2d(values.sum(axis=1)).T
    nozeros += 0.001
    values /= nozeros
    colors = array([where(values[i,:] == max(values[i,:]), 1, 0) for i in range(len(values))])

    # plot quiver field in each direction
    for a, (x, y) in enumerate(environment.allActions):
        plt.quiver(X, Y, values[:,a].reshape(shape)*y, values[:,a].reshape(shape)*x, colors[:,a].reshape(shape), scale=10, color='green', width=0.003)
    plt.gcf().canvas.draw()
    
    print "exploration probability:", explorer.epsilon
        
    
plt.show()
