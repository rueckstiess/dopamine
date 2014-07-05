from __future__ import division
import numpy as np
from dopamine.classifier.classifier import Classifier


class NaiveBayes(Classifier):
    """ NaiveBayes implementation for binary feature vectors with +1 smoothing. """
    
    parametric = False

    def classify(self, inp):
        prob = np.zeros(self.nclasses)
        total_count = np.sum(self.class_count)

        for c in xrange(self.nclasses):
            prob[c] = np.log(self.class_count[c] / total_count)
            for i,v in enumerate(inp):
                if v:
                    prob[c] += np.log(self.feature_count[i, c] / self.class_count[c])
                else:
                    prob[c] += np.log((self.class_count[c] - self.feature_count[i, c]) / self.class_count[c])
        return np.argmax(prob)

        
    def update(self, inp, tgt, imp=1.):
        if imp == 0:
            return

        self.class_count[tgt] += 1

        for i,v in enumerate(inp):
            self.feature_count[i, tgt] += v
    
    def reset(self):
        self.class_count = np.ones(self.nclasses) * self.nclasses
        self.feature_count = np.ones((self.indim, self.nclasses))  # smoothed feature counts
