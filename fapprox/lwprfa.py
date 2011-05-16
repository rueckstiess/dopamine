from dopamine.fapprox.fa import FA
from lwpr import LWPR
import numpy as np
import hashlib
import time
import os

class LWPRFA(FA):
    
    parametric = False

    def reset(self):
        FA.reset(self)
        
        # initialize the LWPR function
        self.lwpr = LWPR(self.indim, self.outdim)     
        self.lwpr.init_D = 10.*np.eye(self.indim)
        self.lwpr.init_alpha = 0.1*np.ones([self.indim, self.indim])
        self.lwpr.meta = True
    
    def predict(self, inp):
        """ predict the output for the given input. """
        inp = self._asFlatArray(inp)
        return self.lwpr.predict(inp)

    def train(self):
        for i, t in self.dataset:
            i = self._asFlatArray(i)
            t = self._asFlatArray(t)
            self.lwpr.update(i, t)


    def __getstate__(self):
        """ required for pickle. removes the lwpr model from the dictionary
            and saves it to file explicitly.
        """
        # create unique hash key for filename and write lwpr to file
        hashkey = hashlib.sha1(str(self.lwpr) + time.ctime()).hexdigest()[:6]
        if not os.path.exists('.lwprmodels'):
            os.makedirs('.lwprmodels')   
        self.filename = '.lwprmodels/lwpr_%s.binary'%hashkey
        self.lwpr.write_binary(self.filename)
        
        # remove lwpr from dictionary and return state
        state = self.__dict__.copy()
        del state['lwpr']
        return state
        
    def __setstate__(self, state):
        """ required for pickle. loads the stored lwpr model explicitly.
        """
        self.__dict__.update(state)
        self.lwpr = LWPR(self.filename)
        