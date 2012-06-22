from numpy import *
import types

class Dataset(object):
    
    def __init__(self, indim, outdim):
        """ Initialize the dataset, create the numpy arrays that store inputs and targets.
        """
        # numpy arrays containing inputs and targets
        self.inputs_ = zeros((0, indim))
        self.targets_ = zeros((0, outdim))
        self.importance_ = zeros([])
        
        # length of the actual data (the arrays might be larger and padded with zeros for efficiency)
        self.length = 0
                
    def clear(self):
        """ clears the dataset, resetting all the arrays to zero length. """
        self.inputs_ = zeros((0, self.inputs_.shape[1]))
        self.targets_ = zeros((0, self.targets_.shape[1]))
        self.importance_ = zeros([])
        self.length = 0
            
    @property
    def indim(self):
        return self.inputs_.shape[1]
    
    @property
    def outdim(self):
        return self.targets_.shape[1]
                
    @property
    def inputs(self):
        """ return only the inputs. """
        return self.inputs_[:self.length, :]
    
    @property
    def targets(self):
        """ return only the targets. """
        return self.targets_[:self.length, :]
                    
    @property
    def importance(self):
        """ return only the importance. """
        return self.importance_[:self.length]

    def append(self, inp, tgt, imp = 1.):
        """ append one input, target (and optionally importance) to the respective arrays. """
        
        # check whether a resize is necessary
        shape = self.inputs_.shape
        if shape[0] <= self.length:
            self.inputs_.resize((2*shape[0]+1, self.inputs_.shape[1]), refcheck=False)
            self.targets_.resize((2*shape[0]+1, self.targets_.shape[1]), refcheck=False)
            self.importance_.resize(2*shape[0]+1, refcheck=False)
        
        # append input, target arrays
        self.inputs_[self.length, :] = asarray(inp)
        self.targets_[self.length, :] = asarray(tgt)
        self.importance_[self.length] = imp
        
        # increase the length counter
        self.length += 1
        
    def extend(self, dataset):
        """ appends the given dataset to this one. """
        if isinstance(dataset, Dataset):
            self.inputs_ = r_[self.inputs, dataset.inputs]
            self.targets_ = r_[self.targets, dataset.targets]
            self.importance_ = r_[self.importance, dataset.importance]
            self.length = len(self.inputs_)
    
    def setArrays(self, inputs, targets, importance=None):
        """ set the inputs, targets and importance arrays manually. they need 
            to be the same length. if importance is not set (default), all 
            importance values are assumed to be 1. 
        """
        if importance == None:
            importance = ones(inputs.shape[0])
        if inputs.shape[0] == targets.shape[0] == importance.shape[0]:
            self.inputs_ = inputs
            self.targets_ = targets
            self.importance_ = importance
            self.length = len(inputs)
        
    def __len__(self):
        """ returns the length (number of samples) of the dataset. """
        return self.length

    def __getitem__(self, key):
        """ Returns a dataset that is a slice of the original dataset
            according to key. key can be an integer (positive or negative)
            or a slice object (represented by x:y notation). 
        """
        ds = Dataset(self.indim, self.outdim)
        if type(key) == types.SliceType:
            ds.setArrays(self.inputs[key,:], self.targets[key,:], self.importance[key])
        elif type(key) == types.IntType:
            ds.append(self.inputs[key,:], self.targets[key,:], self.importance[key])
        return ds

    def __iter__(self):
        """ iterate over dataset and return input and target. """
        for i in xrange(self.length):
            inp = self.inputs[i,:]
            tgt = self.targets[i,:]
            yield(inp, tgt)
    
    def iterate(self, importance=False):
        for i in xrange(self.length):
            inp = self.inputs[i,:]
            tgt = self.targets[i,:]
            imp = self.importance[i]

            if importance:
                yield(inp, tgt, imp)
            else:
                yield(inp, tgt)

    def randomizedSamples(self, importance=False):
        """ iterate over all samples in history (like __iter__) but
            in random order. return input and target for each iteration. """
        for i in random.permutation(xrange(self.length)):
            inp = self.inputs[i,:]
            tgt = self.targets[i,:]
            imp = self.importance[i]

            if importance:
                yield(inp, tgt, imp)
            else:
                yield(inp, tgt)
    
    def __str__(self):
        """ string representation of episode. prints inputs, targets, importance. """
        out = []
        for inp,tgt,imp in self.iterate(True):
            out.append(str(inp) + ' ' + str(tgt) + ' ' + str(imp))
        return "\n".join(out)
        


    
    
    
        
    
        
            
            
    
