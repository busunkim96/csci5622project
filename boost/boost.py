#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.base import clone 
import matplotlib.pyplot as plt
import math

np.random.seed(1234)

class Hanzi:
    """
    Class to store MNIST data
    """

    def __init__(self, location, MNIST=False):

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

        x = np.array(self.test_x)
        nsamples, nx, ny = x.shape
        d2x = x.reshape((nsamples,nx*ny))
        self.test_x = d2x
        #self.test_y = np.array(self.test_y)
        self.test_y = self.convertYs(self.test_y)

        x = np.array(self.train_x)
        nsamples, nx, ny = x.shape
        d2x = x.reshape((nsamples,nx*ny))
        self.train_x = d2x
        #self.train_y = np.array(self.train_y)
        self.train_y = self.convertYs(self.train_y)

    def convertYs(self, y):
        y0 = np.zeros(len(y))
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

class AdaBoost:
    def __init__(self, n_learners=20, base=DecisionTreeClassifier(max_depth=1)):
        """
        Create a new adaboost classifier.
        
        Args:
            n_learners (int, optional): Number of weak learners in classifier.
            base (BaseEstimator, optional): Your general weak learner 

        Attributes:
            base (estimator): Your general weak learner 
            n_learners (int): Number of weak learners in classifier.
            alpha (ndarray): Coefficients on weak learners. 
            learners (list): List of weak learner instances. 
        """
        
        self.n_learners = n_learners 
        self.base = base
        self.alpha = np.zeros(self.n_learners)
        self.learners = []
        
    def fit(self, X_train, y_train):
        """
        Train AdaBoost classifier on data. Sets alphas and learners. 
        
        Args:
            X_train (ndarray): [n_samples x n_features] ndarray of training data   
            y_train (ndarray): [n_samples] ndarray of data 
        """

        w = np.full(len(y_train), 1.0/len(y_train))

        for k in range(self.n_learners):
            h = clone(self.base)
            h.fit(X_train, y_train, sample_weight=w)

            prediction = np.zeros(len(y_train))
            for i in range(len(y_train)):
                prediction[i] = h.predict(X_train[i].reshape(1, -1))

            error = 0.0
            for i in range(len(y_train)):
                if y_train[i] != prediction[i]:
                    error += w[i]
            error = error/np.sum(w)

            if error != 0:
                self.alpha[k] = .5*math.log((1 - error)/error)
            else: 
                print 'Divide by 0 error in alpha'

            for i in range(len(y_train)):
                w[i] = (w[i])*math.exp(-self.alpha[k]*y_train[i]*prediction[i])
            w = w/np.sum(w)

            self.learners.append(h)

    def predict(self, X):
        """
        Adaboost prediction for new data X.
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            
        Returns: 
            [n_samples] ndarray of predicted labels {-1,1}
        """

        prediction = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            for k in range(len(self.learners)):
                prediction[i] += self.alpha[k]*self.learners[k].predict(X[i].reshape(1, -1))
            if prediction[i] < 0:
                prediction[i] = -1
            else:
                prediction[i] = 1
        return prediction
    
    def score(self, X, y):
        """
        Computes prediction accuracy of classifier.  
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            y (ndarray): [n_samples] ndarray of true labels  
            
        Returns: 
            Prediction accuracy (between 0.0 and 1.0).
        """

        correct = 0.0
        for i in range(X.shape[0]):
            if self.predict(X[i].reshape(1, -1)) == y[i]:
                correct += 1
        return correct/X.shape[0]
    
    def staged_score(self, X, y):
        """
        Computes the ensemble score after each iteration of boosting 
        for monitoring purposes, such as to determine the score on a 
        test set after each boost.
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            y (ndarray): [n_samples] ndarray of true labels  
            
        Returns: 
            [n_learners] ndarray of scores 
        """

        scores = np.zeros(self.n_learners)
        allLearners = self.learners
        for k in range(self.n_learners):
            self.learners = allLearners[:(k+1)]
            scores[k] = self.score(X, y)
        return scores

def mnist_digit_show(flatimage, outname=None):

	import matplotlib.pyplot as plt

	image = np.reshape(flatimage, (-1,28))

	plt.matshow(image, cmap=plt.cm.binary)
	plt.xticks([])
	plt.yticks([])
	if outname: 
	    plt.savefig(outname)
	else:
	    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AdaBoost classifier options')
    parser.add_argument('--limit', type=int, default=-1, help="Restrict training to this many examples")
    parser.add_argument('--n_learners', type=int, default=50, help="Number of weak learners to use in boosting")
    parser.add_argument("--MNIST", help="use MNIST set", dest="MNIST", action="store_true")
    args = parser.parse_args()
    
    data = None
    if args.MNIST:
        print "Using MNIST data"
        data = Hanzi("../mnist.pkl.gz", MNIST=True)
    else:
        print "Using CASIA data"
        data = Hanzi("../casia.pkl.gz")

    #for i in [1,2,3]:
        #for j in [5,10,25,50,70,100,500]:
            #print "decision tree ", i, " learner ", j
    clf = AdaBoost(n_learners=100, base=DecisionTreeClassifier(max_depth=1, criterion="entropy"))
    clf.fit(data.train_x, data.train_y)
    print clf.score(data.train_x, data.train_y)
    print clf.score(data.test_x, data.test_y)
