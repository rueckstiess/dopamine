from rllib.environments import MDPMaze
from rllib.agents import QAgent
from rllib.experiments import Experiment
from rllib.adapters import MakeEpisodicAdapter, EpsilonGreedyExplorer

from matplotlib import pyplot as plt
import time

agent = QAgent()
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
    print explorer.epsilon
    plt.pcolor(agent.estimator.values.max(axis=1).reshape(9, 9))
    plt.gcf().canvas.draw()
