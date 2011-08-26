from numpy import *
import types

class Dataset(object):
    
    def __init__(self, indim, outdim):
        """ Initialize the dataset, create the numpy arrays that store inputs and targets.
        """
        # numpy arrays containing inputs and targets
        self.inputs_ = zeros((0, indim))
        self.targets_ = zeros((0, outdim))
        
        # length of the actual data (the arrays might be larger and padded with zeros for efficiency)
        self.length = 0
                
    def clear(self):
        """ clears the dataset, resetting all the arrays to zero length. """
        self.inputs_ = zeros((0, self.inputs_.shape[1]))
        self.targets_ = zeros((0, self.targets_.shape[1]))
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
                    
    def append(self, inp, tgt):
        """ append one input and target to the respective arrays. """
        
        # check whether a resize is necessary
        shape = self.inputs_.shape
        if shape[0] <= self.length:
            self.inputs_.resize((2*shape[0]+1, self.inputs_.shape[1]), refcheck=False)
            self.targets_.resize((2*shape[0]+1, self.targets_.shape[1]), refcheck=False)
        
        # append input, target arrays
        self.inputs_[self.length, :] = asarray(inp)
        self.targets_[self.length, :] = asarray(tgt)
        
        # increase the length counter
        self.length += 1
        
    def extend(self, dataset):
        """ appends the given dataset to this one. """
        if isinstance(dataset, Dataset):
            self.inputs_ = r_[self.inputs, episode.inputs]
            self.targets_ = r_[self.targets, episode.targets]
            self.length = len(self.inputs_)
    
    def setArrays(self, inputs, targets):
        """ set the inputs and targets arrays manually. they need to be the same length. """
        if inputs.shape[0] == targets.shape[0]:
            self.inputs_ = inputs
            self.targets_ = targets
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
            ds.setArrays(self.inputs[key,:], self.targets[key,:])
        elif type(key) == types.IntType:
            ds.append(self.inputs[key,:], self.targets[key,:])
        return ds
    
    def __iter__(self):
        """ iterate over dataset and return input and target. """
        for i in xrange(self.length):
            inp = self.inputs[i,:]
            tgt = self.targets[i,:]
            yield(inp, tgt)
    
    def randomizedSamples(self):
        """ iterate over all samples in history (like __iter__) but
            in random order. return input and target for each iteration. """
        for i in random.permutation(xrange(self.length)):
            inp = self.inputs[i,:]
            tgt = self.targets[i,:]
            yield(inp, tgt)
    
    def __str__(self):
        """ string representation of episode. prints inputs and targets. """
        out = []
        for i,t in self:
            out.append(str(i) + ' ' + str(t))
        return "\n".join(out)
        


    
    
    
        
    
        
            
            
    
