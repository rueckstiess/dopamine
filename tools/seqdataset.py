from dopamine.tools.dataset import Dataset
from numpy import zeros, r_
from numpy import random
import types

class SequenceDataset(object):

    def __init__(self, indim, outdim):
        self.indim = indim
        self.outdim = outdim
        self.sequences_ = [Dataset(indim, outdim)]
        
    # @property
    # def indim(self):
    #     return self.sequences_[0].indim
    # 
    # @property
    # def outdim(self):
    #     return self.sequences_[0].outdim
    
    def clear(self):
        self.sequences_ = [Dataset(self.indim, self.outdim)]
    
    def newSequence(self):
        """ signals the end of the current sequence and prepares the history
            to append new samples to a fresh sequence.
            
            Example:
            seqds => [aaaa bbb]
            seqds.newDataset()
            seqds => [aaaa bbb .]
        """
        if len(self.sequences_) == 0 or len(self.sequences_[-1]) > 0:
            self.sequences_.append(Dataset(self.indim, self.outdim))
        
    def editLastSequence(self):
        """ this reverses the effect of newSequence() and will append
            new samples to the last existing sequence instead. If the
            last sequence is not empty, this call has no effect. 
            
            Example:
            seqds => [aaaa bbb .]
            seqds.newDataset()
            seqds => [aaaa bbb]
        """
        if len(self.sequences) > 0 and len(self.sequences_[-1]) == 0:
            self.sequences_ = self.sequences_[:-1]
    
    def append(self, inp, tgt, imp=1.):
        """ appends samples to the current sequence. If newSequence() was
            called, append will start adding samples to that new sequence.
        """
        self.sequences_[-1].append(inp, tgt, imp)
    
    def appendSequence(self, sequence):
        """ This will append a full sequence to the history rather than
            single samples with append(). If the last sequence was open
            and not finished yet, it will be finished and a new sequence
            starts after the added sequence.
            
            Example (. is the empty sequence):
            
            seqds => [aaaaa bb cccc]
            seqds.appendSequence([ddd])
            seqds => [aaaaa bb cccc ddd .]
        """
        lastseq = self.sequences_[-1]
        if len(lastseq) == 0:
            self.sequences_ = self.sequences_[:-1]
        self.sequences_.append(sequence)
        self.newSequence()
        
    def extend(self, sequence_ds):
        lastseq = self.sequences_[-1]
        if len(lastseq) == 0:
            self.sequences_ = self.sequences_[:-1]
        self.sequences_.extend(sequence_ds)
        if len(lastseq) == 0:
            self.newSequence()
        
    def numTotalSamples(self):
        return sum([len(s) for s in self.sequences])
        
    def pop(self, nonempty=True):
        """ returns and removes the last (by default: non-empty) sequence in 
            the sequence dataset. after this call, sequence dataset will always 
            have a new empty sequence as it's last element onto which new samples 
            are added.
        """
        if not nonempty:
            sequence = self.sequences_.pop()
            self.newSequence()
        
        else:            
            if len(self.sequences) == 0:
                raise IndexError('pop from empty sequence dataset')
            
            sequence = self.sequences[-1]
            self.sequences_ = self.sequences[:-1]
            self.newSequence()
        
        return sequence
    
    def truncate(self, n, newest=True):
        """ truncates the sequence dataset to leave only n sequences. if newest is set
            to True (default), the n most recent sequences are kept, else the n
            oldest sequences are kept. 
        """
        if newest:
            self.sequences_ = self.sequences_[-n:]
        else:
            self.sequences_ = self.sequences_[:n]
         
    def shuffle(self):
        """ shuffles the sequences in random order, then starts a new sequence. """
        self.sequences_ = [self.sequences_[i] for i in random.permutation(len(self.sequences))]
        self.newSequence()
    
    @property
    def sequences(self):
        """ if the last sequence is empty, do not consider it. """
        if len(self.sequences_[-1]) == 0:
            return self.sequences_[:-1]
        else:
            return self.sequences_
    
    @property
    def inputs(self):
        """ return array of all states over all sequences in shape n x indim """
        inp = zeros((0, self.indim))
        for s in self.sequences:
            inp = r_[inp, s.inputs.reshape(len(s), self.indim)]
        return inp
    
    @property
    def targets(self):
        """ return array of all targets over all sequences in shape n x outdim """
        tgts = zeros((0, self.outdim))
        for s in self.sequences:
            tgts = r_[tgts, s.targets.reshape(len(s), self.outdim)]
        return tgts
            
    @property
    def importance(self):
        """ return array of all targets over all sequences in shape n x outdim """
        imp = zeros([])
        for s in self.sequences:
            imp = r_[imp, s.importance]
        return imp
            
    def __len__(self):
        """ returns the number of sequences (empty sequence at the end not considered). """
        return len(self.sequences)
        
    def __iter__(self):
        """ iterates over sequences (empty sequence at the end not considered). """        
        for s in self.sequences:
            yield s
    
    def __getitem__(self, key):
        """ if key is a slice object, return a new SequenceDataset containing the sliced
            sequences. if key is of type int, return that sequence (of type Dataset).
        """
        if type(key) == types.SliceType:
            sds = SequenceDataset(self.indim, self.outdim)
            for s in self.sequences[key]:
                sds.appendSequence(s)
            return sds
        elif type(key) == types.IntType:
            return self.sequences[key]
    
    def __str__(self):
        out = []
        for sequence in self.sequences:
            out.append(str(sequence))
        return "\n\n".join(out)