from dopamine.environments import TestEnvironment
from dopamine.agents import NFQAgent
from dopamine.experiments import Experiment
from dopamine.adapters import EpsilonGreedyExplorer, BoltzmannExplorer

from matplotlib import pyplot as plt
from numpy import *


def plotPolicy(agent):
    plt.clf()
    for s in arange(-1, 1, 0.01):
        s = array([s])
        q0 = agent.estimator.getValue(s, array([0]))
        q1 = agent.estimator.getValue(s, array([1]))
        plt.plot(s, q0, '.r')
        plt.plot(s, q1, '.b')
    plt.gcf().canvas.draw()


# create agent, environment, renderer, experiment
agent = NFQAgent()
environment = TestEnvironment()
experiment = Experiment(environment, agent)

# add normalization adapter
# normalizer = NormalizingAdapter()
# experiment.addAdapter(normalizer)

# add e-greedy exploration
explorer = BoltzmannExplorer(2.0, decay=0.9999)
# explorer = EpsilonGreedyExplorer(0.4, decay=0.9999)
experiment.addAdapter(explorer)

# run 10 episodes to initialize the normalizing adapter
for i in range(10):
    experiment.runEpisode(reset=True)

# print "normalizing:", normalizer.minStates, normalizer.maxStates

agent.forget()

plt.ion()

# run experiment
for i in range(100):
    experiment.runEpisode(reset=True)
    agent.learn()
    
    print "mean rewards:", mean(agent.episode.rewards)
    print "exploration:", explorer.tau
    print agent.episode
    plotPolicy(agent)
    
