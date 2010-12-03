from dopamine.environments import TestEnvironment
from dopamine.agents import FQIAgent
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

    
    # inps = agent.estimator.dataset['input']
    # tgts = agent.estimator.dataset['target'].flatten()
    # 
    # red = where(inps[:,1])[0]
    # blue = where(inps[:,2])[0]
    # 
    # plt.plot(inps[red,0].flatten(), tgts[red], 'sr', alpha=0.5)
    # plt.plot(inps[blue,0].flatten(), tgts[blue], 'sb', alpha=0.5)
    plt.gcf().canvas.draw()

# create agent, environment, renderer, experiment
agent = FQIAgent()
environment = TestEnvironment()
experiment = Experiment(environment, agent)

# add normalization adapter
# normalizer = NormalizingAdapter()
# experiment.addAdapter(normalizer)

# add e-greedy exploration
# explorer = BoltzmannExplorer(2.0, decay=0.999)
explorer = EpsilonGreedyExplorer(0.5, decay=0.999)
experiment.addAdapter(explorer)

# run 10 episodes to initialize the normalizing adapter
for i in range(10):
    experiment.runEpisode(reset=True)

# print "normalizing:", normalizer.minStates, normalizer.maxStates

agent.forget()

plt.ion()

# run experiment
for i in range(1000):
    for i in range(1):
        experiment.runEpisode(reset=True)
    agent.learn()

    print "mean rewards:", mean(agent.episode.rewards)
    print "exploration:", explorer.epsilon
    
    plotPolicy(agent)
    
