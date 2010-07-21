from dopamine.environments import DiscreteCartPoleEnvironment, CartPoleRenderer
from dopamine.agents import FQIAgent, RBFEstimator
from dopamine.agents import LWPREstimator
from dopamine.experiments import Experiment
from dopamine.adapters import EpsilonGreedyExplorer, NormalizingAdapter, IndexingAdapter

from matplotlib import pyplot as plt
from numpy import *
from mpl_toolkits.mplot3d import axes3d

# create agent, environment, renderer, experiment
agent = FQIAgent(estimatorClass=LWPREstimator)
agent.iterations = 1

environment = DiscreteCartPoleEnvironment()
environment.centerCart = False
environment.conditions['actionNum'] = 4
experiment = Experiment(environment, agent)

# cut off last two state dimensions
# indexer = IndexingAdapter([0, 1], None)
# experiment.addAdapter(indexer)

# add normalization adapter
normalizer = NormalizingAdapter()
experiment.addAdapter(normalizer)

# add e-greedy exploration
explorer = EpsilonGreedyExplorer(0.3, 0.9995)
experiment.addAdapter(explorer)

# renderer = CartPoleRenderer()
# environment.renderer = renderer
# renderer.start()

# run experiment

plt.ion()
fig = plt.figure()
ax = axes3d.Axes3D(fig)
frame = None
xgrid,ygrid = mgrid[-1:1:0.1, -12:12:1]

for i in range(100):
    experiment.runEpisodes(1)
    agent.learn()
    # only keep the 10 most recent episodes
    # agent.history.truncate(50, newest=True)

    # agent.forget()
    
    valdata = experiment.evaluateEpisodes(20, visualize=True)
    print "exploration", explorer.epsilon
    print "mean return", mean([sum(v.rewards) for v in valdata])
    print "num episodes", len(agent.history)
    print "num total samples", agent.history.numTotalSamples()
    
        
    # # # model 1
    # zgrid1 = zeros(len(xgrid.flatten()))
    # for i, (x, y) in enumerate(zip(xgrid.flatten(), ygrid.flatten())):
    #     zgrid1[i] = agent.estimator.lwpr.predict(array([x, y, 0, 1]))
    # 
    # # model 2
    # zgrid2 = zeros(len(xgrid.flatten()))
    # for i, (x, y) in enumerate(zip(xgrid.flatten(), ygrid.flatten())):
    #     zgrid2[i] = agent.estimator.lwpr.predict(array([x, y, 1, 0]))
    # 
    # if frame:
    #     ax.collections.remove(frame)
    # 
    # frame = ax.plot_surface(xgrid, ygrid, zgrid1.reshape(xgrid.shape) - zgrid2.reshape(xgrid.shape), rstride=1, cstride=1, cmap=plt.cm.jet, antialiased=True)
    # plt.draw()

plt.show()

