from dopamine.environments import MDPMaze
from dopamine.agents import QAgent, SARSAAgent
from dopamine.experiments import Experiment
from dopamine.adapters import MakeEpisodicAdapter, EpsilonGreedyExplorer

from matplotlib import pyplot as plt
import time


agent = QAgent()
# agent = SARSAAgent()

environment = MDPMaze()
experiment = Experiment(environment, agent)
experiment.addAdapter(MakeEpisodicAdapter(100))

explorer = EpsilonGreedyExplorer(0.3, 0.9999)
experiment.addAdapter(explorer)

plt.ion()
plt.gray()

for i in range(100):
    experiment.runEpisode(reset=True)
    agent.learn()
    agent.forget()
    plt.pcolor(agent.estimator.values.max(axis=1).reshape(9, 9))
    plt.gcf().canvas.draw()
