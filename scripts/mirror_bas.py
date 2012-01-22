from dopamine.environments import MirrorEnvironment
from dopamine.agents import APIAgent
from dopamine.experiments import Experiment
from dopamine.adapters import EpsilonGreedyExplorer, BinaryActionSearchAdapter
from dopamine.fapprox import *
import numpy as np

# create agent, environment, renderer, experiment
agent = APIAgent(faClass=BLinReg, resetFA=False, vectorblock=False)
agent.gamma = 1.0
environment = MirrorEnvironment()
experiment = Experiment(environment, agent)

# add bas adapter
bas = BinaryActionSearchAdapter(3., 4., 4)
experiment.addAdapter(bas)

# add e-greedy exploration
explorer = EpsilonGreedyExplorer(0.5, decay=0.99995)
experiment.addAdapter(explorer)

# run experiment
for i in range(1000):
    experiment.runEpisodes(1000)
    agent.learn()

    valdata = experiment.evaluateEpisodes(1000)
    print "mean rewards:", np.mean([sum(e.rewards) for e in valdata]), "exploration:", explorer.epsilon
    # print "exploration:", explorer.epsilon
    
    agent.forget()
