from dopamine.environments import DiscreteCartPoleEnvironment, CartPoleRenderer
from dopamine.agents import FQIAgent, RBFEstimator, NNEstimator, RBFOnlineEstimator, LWPREstimator
from dopamine.experiments import Experiment
from dopamine.adapters import EpsilonGreedyExplorer, NormalizingAdapter, IndexingAdapter

from matplotlib import pyplot as plt
from numpy import *
import os, optparse, cPickle


def parse_opt():
    """ parses the command line options for different settings. """
    optparser = optparse.OptionParser()
    optparser.add_option('-p', '--play',
        action='store_true', dest='play', default=False, 
        help="play in renderer")

    options, args = optparser.parse_args()
    return options, args


options, args = parse_opt()

# create agent, environment, renderer, experiment
if options.play and os.path.exists('cart_play.saved'):
    print 'loading agent...'
    f = open('cart_play.saved', 'r')
    agent = cPickle.load(f)
else:
    print "no saved agent found. start from scratch."
    agent = FQIAgent(estimatorClass=RBFEstimator)
    options.play = False

agent.iterations = 1
environment = DiscreteCartPoleEnvironment()
environment.conditions['actionNum'] = 2
environment.centerCart = False
experiment = Experiment(environment, agent)

# cut off last two state dimensions
indexer = IndexingAdapter([0, 1], None)
experiment.addAdapter(indexer)

# add normalization adapter
normalizer = NormalizingAdapter(normalizeStates=[(-0.8, 0.8), (-12., 12.)])
experiment.addAdapter(normalizer)

if options.play:
    renderer = CartPoleRenderer()
    environment.renderer = renderer
    renderer.start()
    print "10 evaluation trials:"
    valdata = experiment.evaluateEpisodes(10, visualize=False)
    mean_return = mean([sum(v.rewards) for v in valdata])
    print "mean return", mean_return
    raise SystemExit

# add e-greedy exploration
explorer = EpsilonGreedyExplorer(0.3, 0.99995)
experiment.addAdapter(explorer)
    
if os.path.exists('cart_play.saved'):
    os.remove('cart_play.saved')
      
# run experiment
for i in range(1000):
    valdata = experiment.evaluateEpisodes(20, visualize=True)
    mean_return = mean([sum(v.rewards) for v in valdata])

    experiment.runEpisodes(1)
    agent.learn()
    # agent.history.truncate(20)
    # agent.forget()
    
    # save file after each learning step
    f = open('cart_play.saved', 'w')
    cPickle.dump(agent, f)
    f.close()
    
    print normalizer.minStates, normalizer.maxStates
    print "params", agent.estimator.models[0].W
    print "exploration", explorer.epsilon
    print "mean return", mean_return
    print "num episodes", len(agent.history)
    print "num total samples", agent.history.numTotalSamples()
