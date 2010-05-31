from dopamine.environments import TestEnvironment
from dopamine.environments import DiscreteCartPoleEnvironment, CartPoleRenderer

from dopamine.agents import QAgent, SARSAAgent
from dopamine.experiments import Experiment
from dopamine.adapters import EpsilonGreedyExplorer, VQDiscretizationAdapter

from matplotlib import pyplot as plt
from numpy import *
import time


# create agent, environment, renderer, experiment
agent = SARSAAgent()
environment = DiscreteCartPoleEnvironment()

experiment = Experiment(environment, agent)

# add discretization adapter
discretizer = VQDiscretizationAdapter(30)
experiment.addAdapter(discretizer)

# add e-greedy exploration
explorer = EpsilonGreedyExplorer(0.3, decay=0.999)
experiment.addAdapter(explorer)

# run 10 episodes to initialize the adapter
print "pretraining..."
for i in range(100):
    print i
    experiment.runEpisode(reset=True)

discretizer.sampleClusters()
discretizer.adaptClusters()

for i in range(len(discretizer.stateVectors)):
    plt.text(discretizer.stateVectors[i,0], discretizer.stateVectors[i,1], "%i"%i, bbox=dict(facecolor='green', alpha=0.5))

plt.xlim(-2.5, 2.5)
plt.ylim(-10, 10)

agent.forget()
explorer.epsilon = 0.3
# renderer = CartPoleRenderer()
# environment.renderer = renderer
# renderer.start()

# run experiment
for i in range(1000):
    experiment.runEpisode(reset=True)
    # discretizer.adaptClusters()
    agent.learn()
        
    print "sum rewards:", sum(agent.episode.rewards)
    print "exploration:", explorer.epsilon
