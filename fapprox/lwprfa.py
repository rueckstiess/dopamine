from dopamine.fapprox.fa import FA
from lwpr import LWPR
import numpy as np

class LWPRFA(FA):
    
    parametric = False

    def reset(self):
        FA.reset(self)
        
        # initialize the LWPR function
        self.lwpr = LWPR(self.indim, self.outdim)     
        self.lwpr.init_D = 100.*np.eye(self.indim)
        self.lwpr.init_alpha = 0.1*np.ones([self.indim, self.indim])
        self.lwpr.meta = True
    
    def predict(self, inp):
        """ predict the output for the given input. """
        inp = self._asFlatArray(inp)
        return self.lwpr.predict(inp)

    def learn(self):
        for i, t in self.dataset:
            i = self._asFlatArray(i)
            t = self._asFlatArray(t)
            self.lwpr.update(i, t)
