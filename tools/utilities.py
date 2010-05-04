from numpy import zeros

def abstractMethod():
    """ This should be called when an abstract method is called that should have been 
    implemented by a subclass. It should not be called in situations where no implementation
    (i.e. a 'pass' behavior) is acceptable. """
    raise NotImplementedError('Method not implemented!')


def one_to_n(val, maxval):
    """ Returns a 1-in-n binary encoding of a non-negative integer. """
    a = zeros(int(maxval), float)
    a[int(val)] = 1.
    return a


def n_to_one(arr):
    """ Returns the reverse of a 1-in-n binary encoding. """
    return where(arr == 1)[0][0]