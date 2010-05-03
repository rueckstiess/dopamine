""" Test all .txt files in ./doctests folder. """

import doctest
import os

for (paths, dirs, files) in os.walk('.'):
    for f in files:
        if f.endswith('.txt'):
            print "testing %s"% os.path.join(paths, f)
            doctest.testfile(os.path.join(paths, f))
            