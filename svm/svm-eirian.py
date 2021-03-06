#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from collections import Counter, defaultdict
import sys

import random
import numpy
from numpy import median
from sklearn.svm import SVC
#from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from skimage.color import rgb2gray
from skimage import io

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing

class Numbers:
    """
    Class to store CASIA data
    """

    def __init__(self, location, MNIST=False, standardize=False):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import cPickle, gzip

        if MNIST:
            # MNIST set
            # Load the dataset
            f = gzip.open(location, 'rb')
            train_set, valid_set, test_set = cPickle.load(f)

            self.train_x, self.train_y = train_set
            #self.test_x, self.test_y = valid_set
            self.test_x, self.test_y = test_set
            if standardize:
                scaler = preprocessing.StandardScaler().fit(self.train_x)
                scaler.transform(self.train_x)
                scaler.transform(self.test_x)
            f.close()
            self.e_scale()
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
        #self.e_scale()

    def e_scale(self):

        # this one doesn't seem to make any statistical difference
        '''
        rs = preprocessing.Imputer(strategy='median', copy=False)
        self.train_x = rs.fit_transform(self.train_x, self.train_y)
        self.test_x = rs.transform(self.test_x)
        '''

        # so, preprocessing schemes are quite different for the data sets
        rs = preprocessing.StandardScaler()
        # polynomial features *might* perform better than OneHotEncoding, but my laptop can't handle it
        ####rs = preprocessing.PolynomialFeatures()
        self.train_x = rs.fit_transform(self.train_x, self.train_y)
        self.test_x = rs.transform(self.test_x)

        '''
        rs = preprocessing.KernelCenterer()
        self.train_x = rs.fit_transform(self.train_x.toarray(), self.train_y)
        self.test_x = rs.transform(self.test_x.toarray())
        '''

        if standardize:
            scaler = preprocessing.StandardScaler().fit(self.train_x)
            scaler.transform(self.train_x)
            scaler.transform(self.test_x)

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


def performKFold(data, k, limit=None):
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    kf.get_n_splits(data.train_x)
    correctpc = []

    if limit > 0:
        data.train_x = data.train_x[:args.limit]
        data.train_y = data.train_y[:args.limit]

    for train_index, test_index in kf.split(data.train_x, data.train_y):
        # XXX tuning parameters

        # XXX coef0 w/poly kernel good grid search configurable
        # XXX C also good for grid search
        # XXX degree also, given the poly kernel

        # TODO enable probability
        # XXX XXX XXX below is some tuning for CASIA data. Should do tuning for MNIST specifically
        # really need to do grid search to find best parameter values
        svm = SVC(C=3.0, kernel='poly', degree=3, gamma='auto', coef0=2.0, shrinking=True, probability=False, tol=0.001, cache_size=400, class_weight=None, max_iter=-1, decision_function_shape="ovr")
        #svm = SVC(C=3.0, kernel='poly', degree=3, gamma='auto', coef0=2.0, shrinking=True, probability=False, tol=0.001, cache_size=400, class_weight="balanced", max_iter=-1, decision_function_shape="ovr")
        #svm = SVC(kernel='linear', decision_function_shape="ovo", C=3.0)

        svm = svm.fit(data.train_x[train_index], data.train_y[train_index])

        correct = svm.score(data.test_x, data.test_y)
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
    parser.add_argument("--standardize", help="standardize features using sklearn preprocessing", dest="standardize", action="store_true")
    args = parser.parse_args()

    data = None
    if args.MNIST:
        print "Using MNIST data"
        data = Numbers("../mnist.pkl.gz", MNIST=True, standardize=args.standardize)
    else:
        print "Using CASIA data"
        data = Numbers("../casia.pkl.gz", standardize=args.standardize)
    knn = None


    if args.kfold:
        data = performKFold(data, args.k, args.limit)

    else:
        print "!!! USING TEST DATA !!!" # not true for MNIST, enable test data when ready
        # XXX tuning parameters
        svm = SVC(C=3.0, kernel='poly', degree=3, gamma='auto', coef0=2.0, shrinking=True, probability=False, tol=0.001, cache_size=400, class_weight=None, max_iter=-1, decision_function_shape="ovr")

        if args.limit > 0:
            print("Data limit: %i" % args.limit)
            svm = svm.fit(data.train_x[:args.limit], data.train_y[:args.limit])
        else:
            svm = svm.fit(data.train_x, data.train_y)

        print("Done loading data")

        correct = svm.score(data.test_x, data.test_y)
        print("Accuracy: %f" % correct)
