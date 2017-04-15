#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from collections import Counter, defaultdict
import sys

import random
import numpy
from numpy import median
from sklearn.neighbors import BallTree, NearestNeighbors, KDTree
from sklearn.ensemble import AdaBoostClassifier
#from sklearn.grid_search import GridSearchCV
from skimage.color import rgb2gray
from skimage import io

from sklearn.model_selection import KFold, StratifiedKFold

class Numbers:
    """
    Class to store CASIA data
    """

    def __init__(self, location, MNIST=False):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import cPickle, gzip

        if MNIST:
            # MNIST set
            # Load the dataset
            f = gzip.open(location, 'rb')
            train_set, valid_set, test_set = cPickle.load(f)

            self.train_x, self.train_y = train_set
            self.test_x, self.test_y = valid_set
            f.close()
            return

        # CASIA otherwise
        f = gzip.open(location, 'rb')
        train_set, test_set = cPickle.load(f)
        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = test_set

        x = numpy.array(self.test_x)
        nsamples, nx, ny = x.shape
        d2x = x.reshape((nsamples,nx*ny))
        self.test_x = d2x
        #self.test_y = numpy.array(self.test_y)
        self.test_y = self.convertYs(self.test_y)

        x = numpy.array(self.train_x)
        nsamples, nx, ny = x.shape
        d2x = x.reshape((nsamples,nx*ny))
        self.train_x = d2x
        #self.train_y = numpy.array(self.train_y)
        self.train_y = self.convertYs(self.train_y)

    def convertYs(self, y):
        y0 = numpy.zeros(len(y))
        for i, item in enumerate(y):
            if item == "零":
                y0[i] = 0
            elif item == "一":
                y0[i] = 1
            elif item == "二":
                y0[i] = 2
            elif item == "三":
                y0[i] = 3
            elif item == "四":
                y0[i] = 4
            elif item == "五":
                y0[i] = 5
            elif item == "六":
                y0[i] = 6
            elif item == "七":
                y0[i] = 7
            elif item == "八":
                y0[i] = 8
            elif item == "九":
                y0[i] = 9
            elif item == "十": 
                y0[i] = 10

        return y0


def confusion_matrix(self, test_x, test_y):
    """
    Given a matrix of test examples and labels, compute the confusion
    matrix for the current classifier.  Should return a dictionary of
    dictionaries where d[ii][jj] is the number of times an example
    with true label ii was labeled as jj.

    :param test_x: Test data representation
    :param test_y: Test data answers
    """

    d = defaultdict(dict)
    data_index = 0
    test_x = self.predict(test_x)
    for xx, yy in zip(test_x, test_y):
        try:
            d[yy][test_x[xx]] += 1
        except KeyError:
            d[yy][test_x[xx]] = 1
        data_index += 1
        if data_index % 100 == 0:
            #print("%i/%i for confusion matrix" % (data_index, len(test_x)))
            pass
    return d

def accuracy(confusion_matrix):
    """
    Given a confusion matrix, compute the accuracy of the underlying classifier.
    """

    # You do not need to modify this function

    total = 0
    correct = 0
    for ii in confusion_matrix:
        total += sum(confusion_matrix[ii].values())
        correct += confusion_matrix[ii].get(ii, 0)

    if total:
        return float(correct) / float(total)
    else:
        return 0.0

def performKFold(data, k, limit=None):
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    kf.get_n_splits(data.train_x)
    correctpc = []

    if limit > 0:
        data.train_x = data.train_x[:args.limit]
        data.train_y = data.train_y[:args.limit]

    for train_index, test_index in kf.split(data.train_x, data.train_y):
        # XXX tuning parameters
        ada = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm="SAMME.R", random_state=None)
        ada = ada.fit(data.train_x[train_index], data.train_y[train_index])

        #confusion = ada.confusion_matrix(data.train_x[test_index], data.train_y[test_index])

        #correct = ada.accuracy(confusion)
        correct = ada.score(data.test_x, data.test_y)
        correctpc.append(correct)
        print "\t% right: ", correct

    print "overall %: ", numpy.mean(correctpc)

def performGridSearch(data, limit=None):
    if limit > 0:
        data.train_x = data.train_x[:args.limit]
        data.train_y = data.train_y[:args.limit]

    #TBD: add fit() and predict() methods

    pass
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest points to use")
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    parser.add_argument("--kfold", help="use k-folds instead of making predictions", dest="kfold", action="store_true")
    parser.add_argument("--MNIST", help="use MNIST set", dest="MNIST", action="store_true")
    parser.add_argument("--grid", help="use a GridSearchCV. This will ause the --kfold argument to be ignored.", dest="grid", action="store_true")

    args = parser.parse_args()

    data = None
    if args.MNIST:
        print "Using MNIST data"
        data = Numbers("../mnist.pkl.gz", MNIST=True)
    else:
        print "Using CASIA data"
        data = Numbers("../casia.pkl.gz")
    knn = None

    if args.kfold:
        data = performKFold(data, args.k, args.limit)

    else:
        print "!!! USING TEST DATA !!!" # not true for MNIST, enable test data when ready
        # XXX tuning parameters
        ada = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm="SAMME.R", random_state=None)

        if args.limit > 0:
            print("Data limit: %i" % args.limit)
            ada = ada.fit(data.train_x[:args.limit], data.train_y[:args.limit])
        else:
            ada = ada.fit(data.train_x, data.train_y)

        print("Done loading data")

        confusion = confusion_matrix(ada, data.test_x, data.test_y)
        uIndex = 11
        if args.MNIST:
            uIndex = 10
        print("\t" + "\t".join(str(x) for x in xrange(uIndex)))
        print("".join(["-"] * 90))
        for ii in xrange(uIndex):
            print("%i:\t" % ii + "\t".join(str(confusion[ii].get(x, 0))
                                       for x in xrange(uIndex)))
        print("Accuracy: %f" % accuracy(confusion))
