#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from collections import Counter, defaultdict
import sys

import random
import numpy
from numpy import median
from sklearn.neighbors import BallTree, NearestNeighbors, KDTree
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


class Knearest:
    """
    kNN classifier
    """

    def __init__(self, x, y, k=5):
        """
        Creates a kNN instance

        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """

        # XXX TODO tuning parameters
        self._kdtree = BallTree(x)
        # XXX try increasing p
        #self._kdtree = BallTree(x, metric="minkowski", p=3) # slower but works good.
        self._y = y
        self._k = k

    def majority(self, item_indices):
        """
        Given the indices of training examples, return the majority label.  If
        there's a tie, return the median of the majority labels (as implemented 
        in numpy).

        :param item_indices: The indices of the k nearest neighbors
        """
        assert len(item_indices) == self._k, "Did not get k inputs"

        #print type(self._y)
        #print self._y[item_indices]
        #items = [self._y[x] for x in item_indices]
        #majority = [x[0] for x in Counter(items).most_common() if x[1] == max(Counter(self._y[item_indices]).values())]
        #majority = [x[0] for x in Counter(self._y[item_indices]).most_common() if x[1] == max(Counter(self._y[item_indices]).values())]
        #return median(majority)

        # I get better accuracy when I use my craptastic majority function below



        # get the items from the indices
        items = [self._y[x] for x in item_indices]
        '''
        print "items: "
        for item in items:
            print item
        '''
        
        # now count occurrences
        dCount = {}
        for item in items:
            if item in dCount:
                dCount[item] = dCount[item] + 1
            else:
                dCount[item] = 1
        #print "dCount: "
        #print dCount
        sItems = list(set(items))
        topA = None
        topB = None
        count = 0

        # yay loops
        for i in xrange(0, len(sItems)):
            if dCount[sItems[i]] > count:
                topA = sItems[i]
                count = dCount[sItems[i]]
            elif dCount[sItems[i]] == count:
                topB = sItems[i]

        majority = topA
        #'''
        if topB != None:
            #majority = numpy.median([topA, topB]) # not what he wanted apparently
            majority = numpy.median(items)
            return majority
        #'''


        #self._y = numpy.array(self._y)
        index = numpy.where(numpy.array(self._y) == majority)
        #print "self._y: "
        #print self._y
        index = index[0]
        #print "index: "
        #print index

        return self._y[index[0]]

    def classify(self, example):
        """
        Given an example, classify the example.

        :param example: A representation of an example in the same
        format as training data
        """

        # Finish this function to find the k closest points, query the
        # majority function, and return the predicted label.
        # Again, the current return value is a placeholder 
        # and definitely needs to be changed. 

        # k == 3 seems to be good
        dist, ind = self._kdtree.query([example], k=self._k) 
        #dist, ind = self._kdtree.query_radius([example], r=self._k, return_distance=True) 

        maj = self.majority(ind[0])
        return maj

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
        for xx, yy in zip(test_x, test_y):
            try:
                d[yy][self.classify(xx)] += 1
            except KeyError:
                d[yy][self.classify(xx)] = 1
            data_index += 1
            if data_index % 100 == 0:
                #print("%i/%i for confusion matrix" % (data_index, len(test_x)))
                pass
        return d

    @staticmethod
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

    for train_index, test_index in kf.split(data.train_x, data.train_y):
        knn = None
        if limit > 0:
            knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit], k)
        else:
            knn = Knearest(data.train_x[train_index], data.train_y[train_index], k)

        confusion = knn.confusion_matrix(data.train_x[test_index], data.train_y[test_index])

        correct = knn.accuracy(confusion)
        correctpc.append(correct)
        print "% right: ", correct

    print "overall %: ", numpy.mean(correctpc)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest points to use")
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    parser.add_argument("--kfold", help="use k-folds instead of making predictions",
                        type=bool, default=False, required=False)
    parser.add_argument("--MNIST", help="use MNIST set", dest="MNIST", action="store_true")

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
        print "!!! USING TEST DATA !!!"
        if args.limit > 0:
            print("Data limit: %i" % args.limit)
            knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit],
                args.k)
        else:
            knn = Knearest(data.train_x, data.train_y, args.k)
 
        print("Done loading data")

        confusion = knn.confusion_matrix(data.test_x, data.test_y)
        uIndex = 11
        if args.MNIST:
            uIndex = 10
        print("\t" + "\t".join(str(x) for x in xrange(uIndex)))
        print("".join(["-"] * 90))
        for ii in xrange(uIndex):
            print("%i:\t" % ii + "\t".join(str(confusion[ii].get(x, 0))
                                       for x in xrange(uIndex)))
        print("Accuracy: %f" % knn.accuracy(confusion))
