from dopamine.environments import CartPoleEnvironment, CartPoleRenderer
from dopamine.agents import ReinforceAgent, NNController, RBFController, LinearController
from dopamine.adapters import IndexingAdapter, NormalizingAdapter, GaussianExplorer, StateDependentExplorer
from dopamine.experiments import Experiment
from numpy import *

environment = CartPoleEnvironment(maxSteps=100)
environment.centerCart = False
agent = ReinforceAgent(controllerClass=RBFController)
experiment = Experiment(environment, agent)

# cut off last two state dimensions
indexer = IndexingAdapter([0, 1], None)
experiment.addAdapter(indexer)

# add normalization adapter
normalizer = NormalizingAdapter(scaleActions=[(-50, 50)])
experiment.addAdapter(normalizer)

# add gaussian explorer
explorer = GaussianExplorer(sigma=0.2)
# explorer = StateDependentExplorer(sigma=0.2)
experiment.addAdapter(explorer)

# run experiment
for i in range(5000):
    experiment.runEpisodes(50)
    agent.learn()
    agent.forget()

    valdata = experiment.evaluateEpisodes(20)
    print "mean return", mean([sum(v.rewards) for v in valdata])
    print "avg. episode length", mean([len(v) for v in valdata])
    print "exploration variance", explorer.sigma
    
plt.show()