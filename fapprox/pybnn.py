# this class uses the PyBrain library, which can be
# found under http://www.pybrain.org.

from numpy import *
from dopamine.fapprox.fa import FA

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers.rprop import RPropMinusTrainer, BackpropTrainer

class PyBrainNN(FA):
    
    parametric = True
    
    def predict(self, inp):
        inp = self._asFlatArray(inp)
        return self._asFlatArray(self.network.activate(inp))

    def update(self, inp, tgt):
        FA.update(self, inp, tgt)
        
        inp = self._asFlatArray(inp)
        tgt = self._asFlatArray(tgt)
        self.pybdataset.addSample(inp, tgt)

    def reset(self):
        FA.reset(self)
        
        # self.network = buildNetwork(self.indim, 2*(self.indim+self.outdim), self.outdim)
        self.network = buildNetwork(self.indim, self.outdim, bias=True)
        self.network._setParameters(random.normal(0, 0.1, self.network.params.shape))
        self.pybdataset = SupervisedDataSet(self.indim, self.outdim)

    def train(self):
        if len(self.pybdataset) == 0:
            return
        # train module with backprop/rprop on dataset
        trainer = RPropMinusTrainer(self.network, dataset=self.pybdataset, batchlearning=True, verbose=False)
        # trainer = BackpropTrainer(self.network, dataset=self.pybdataset, batchlearning=True, verbose=True)
        trainer.trainEpochs(100)        

    def dOutdTheta(self, inp, outp):
        self.network.reset()
        self.network.activate(inp)
        self.network.backActivate(outp)
        return self.network.derivs

    def _getParameters(self):
        return self._asFlatArray(self.network.params)
        
    def _setParameters(self, parameters):
        self.network._setParameters(parameters)
        
    parameters = property(_getParameters, _setParameters)
