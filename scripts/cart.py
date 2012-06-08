from dopamine.environments import DiscreteCartPoleEnvironment, CartPoleRenderer
from dopamine.agents import *
from dopamine.fa import *
from dopamine.experiments import Experiment
from dopamine.adapters import BoltzmannExplorer, EpsilonGreedyExplorer, NormalizingAdapter, IndexingAdapter

from matplotlib import pyplot as plt
from numpy import *
import sys

agentClass = FQIAgent
estimatorFAClass = RBF

if len(sys.argv) > 1:
    agentClass = eval(sys.argv[1] + 'Agent')

if len(sys.argv) > 2:
    estimatorFAClass = eval(sys.argv[2])


# create agent, environment, renderer, experiment
agent = agentClass(estimatorFAClass=estimatorFAClass)
agent.presentations = 1
environment = DiscreteCartPoleEnvironment()
environment.conditions['actionNum'] = 2
environment.centerCart = False
experiment = Experiment(environment, agent)

# cut off last two state dimensions
indexer = IndexingAdapter([0, 1], None)
experiment.addAdapter(indexer)

# add normalization adapter
normalizer = NormalizingAdapter()
experiment.addAdapter(normalizer)

# add e-greedy exploration
explorer = EpsilonGreedyExplorer(0.5, episodeCount=500)
experiment.addAdapter(explorer)
    
# run experiment
for i in range(1000):
    experiment.runEpisodes(1)
    agent.learn()
    
    valdata = experiment.evaluateEpisodes(10, visualize=True)
    mean_return = mean([sum(v.rewards) for v in valdata])
    print "exploration", explorer.epsilon
    print "mean return", mean_return
    print "num episodes", len(agent.history)
    print "num total samples", agent.history.numTotalSamples()
