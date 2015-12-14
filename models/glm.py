# -*- coding: utf-8 -*-
import numpy as np
import sys
import math
import copy
import random
from sklearn import linear_model
import sklearn.metrics as metrics
from scipy.special import expit
import scipy


''' Base level model interface.  Has a train() and test() method '''
class Model(object):
    def train(self, data, epochs=None):
        pass
    def test(self, data):
        y = [data[i][1] for i in range(len(data))]       
        y_hat = [self.predictor(x_i) for x_i, y_i in data] 
        print metrics.classification_report(y, y_hat)

    def test_confidence(self, data):
        y = [data[i][1] for i in range(len(data))]       
        y_hat, confidence = [self.predictor(x_i) for x_i, y_i in data] 
        #print metrics.classification_report(y, y_hat)

    # If the output confidence has a probability measure, then use it.
    def output_confidence(self, x):
        pass

    # implements leave-k-out cross-validation on the train set.
    def cv(self, trainset, k=10, fold=False):
        n = len(trainset) / k

        y = []
        y_hat = []
        y_confid = []
        random.shuffle(trainset)

        for i in range(n):
            traincopy = copy.deepcopy(trainset)
            validate = traincopy[ i * k : (i + 1)*k]
            traincopy = traincopy[ :i*k] + trainset[k*(i + 1):]
            
            self.train(traincopy)

            y += [validate[i][1] for i in range(k)]
            y_hat +=  [self.predictor(x_i) for x_i, y_i in validate]
            y_confid += [ (x_i, self.output_confidence(x_i, y_i)) for x_i, y_i in validate]

            # Recreate the train.   
            traincopy = traincopy[ :i*k] + validate + traincopy[i*k :] 

        print metrics.classification_report(y, y_hat)
        with open('confidence.txt', 'w+') as f:
            for predicted, real, conf in zip(y, y_hat, y_confid):
                if predicted != real:
                    f.write("{0}|{1}\n".format(conf[1], " ".join([str(c) for c in conf[0]]) ) )


# A classifier is something with a train and predict method.
class Classifier(Model):
    
    def train(self, data, epochs=None):
        # Calculate objectives
        if self.epochs is None:
            self.epochs = epochs if epochs else 20
        
        num_feat = data[0][0].size
        n = len(data)


        self.theta = np.zeros(num_feat)
        # print self.theta

        for it in range(self.epochs):        
            alph = self.alpha(it)
            error = 0
            for x, y in data:
                if self.predictor(x) != y:
                    error += 1
                if self.shouldUpdate(x, y):
                    self.theta -= alph * self.df_theta(x, y)
            print "Iter {0} Error: {1}".format(it, error / (1.0 * n))
        return self.theta

    def alpha(self, iter=1):        
        return 1.0


    ''' Runs the class predictor and outputs a label.  Override this in an subclass.'''
    def predictor(self, x):
        pass
    
    ''' Implements the standard online predictor logic '''
    def shouldUpdate(self, x, y):
        return True

    ''' Computes the gradient of the loss with respect to theta. '''
    def df_theta(self, x, y):
        pass

''' 
    Wrapper around an adagrad learner.  
    Uses the gradient history to minimize regret.
'''
class AdagradClassifier(Classifier):
    
    def __init__(self, g_bottom=1e-2):
        self.bottom = g_bottom

    ''' Uses Adagrad to optimize a stochastic gradient descent operation
        efficiently. If full = True, does the full matrix regret minimization 
        procedure, or automatically if the feature space is very small.  Otherwise,
        uses the diagonal regret minimization procedure.
    ''' 
    def init_Gt(self):
        self.full = False
        G_t =  self.bottom * np.ones(self.num_params())
        return G_t

    def num_params():
        pass
    
    def train(self, data, epochs=None):
        # Calculate objectives
        num_feat = data[0][0].size

        if self.epochs is None:
            self.epochs = epochs if epochs else 20
        self.num_params = lambda: num_feat 
        # Introspection, so that we don't redefine theta if it doesn't exist

        self.theta = np.zeros(num_feat)
        self.theta[-1] = 1e-4 # Bias
        
        eta = self.alpha(1)
        numex = len(data)


        G_t = self.init_Gt()

        for it in range(self.epochs):
            error = 0
            for x, y in data:

                if y != self.predictor(x) :
                    error += 1
                if self.shouldUpdate(x, y):
                    g_t = self.df_theta(x, y)
                    print g_t
                    update = eta * g_t / G_t
                    G_t += (g_t * g_t)
                    self.theta -= update 
        
            print "Iter {0} Error: {1}".format(it, error / (1.0 * numex))        
           
        return self.theta
''' Implements Logistic Regression with l2 regularization '''
class LogisticClassifier(AdagradClassifier):
    def __init__(self, epochs=None, reg=0.001, alph=0.1, threshold=0.5):
        super(AdagradClassifier, self).__init__()
        self.bottom= 1e-2
        self.epochs = epochs if epochs else 20
        self.alpha = lambda it: alph
        self.reg = reg 
        self.threshold = threshold

    def predictor(self, x_i):
        return 1 if self.output_confidence(x_i) > self.threshold else 0
    
    def df_theta(self, x_i, y_i):
        prod = self.logistic(np.dot(self.theta, x_i))
        print prod
        print y_i
        print x_i

        return (prod - y_i) * x_i + self.reg * self.theta
    
    def logistic(self, z):
        return expit(z)

    # Override Model implementation
    def output_confidence(self, x):
        return self.logistic(np.dot(self.theta, x))
       
''' Least Squares Regression with l2 regularization '''
class LeastSquaresClassifier(Classifier):
    def __init__(self, epochs=None, reg=0.001):

        self.epochs = epochs if epochs else 50
        self.alpha = lambda it: 0.03
        self.reg = reg
    
    def predictor(self, x_i):
        return 1 if np.dot(self.theta, x_i) >= 0.0 else 0
    
    def df_theta(self, x_i, y_i):
        return (np.dot(self.theta, x_i) - y_i) * x_i + self.reg * self.theta

''' Hinge loss with l2 regularization.'''
class HingeLossClassifier(Classifier):
    def __init__(self, epochs=None, reg=0.001, alpha = (lambda iter: 0.02 / np.sqrt(1 + iter))):
        self.epochs = epochs if epochs else 20
        self.reg = reg
        self.alpha = alpha
        
    def predictor(self, x_i):
        return 1 if np.dot(self.theta, x_i) >= 0.0 else -1

    def shouldUpdate(self, x_i, y_i):
        margin = y_i * np.dot(self.theta, x_i)
        return max(0, 1 - margin) >= 1
        
    def df_theta(self, x_i, y_i):
        dotprod = np.dot(self.theta, x_i)
        if dotprod * y_i < 1:
            return -1.0 * x_i * y_i + self.reg * self.theta
        else:
            return self.reg * self.theta

    # Outputs a tightened sigmoid
    def output_confidence(self, x_i, y_i):

        margin = y_i * np.dot(self.theta, x_i)
        return expit(margin * 1e5 + 0.5)
        
""" Implements linear regression.  Not bad. """

''' Takes a data set and turns it into a set of vocabulary features '''
def vocabulary(data):
    vocab = {}
    count = 0
    for ex in data:
        for word in ex.split(" "):
            if word not in vocab:
                vocab[word] = count
                count += 1
    print len(vocab)
    return vocab


''' Takes a list of data examples and encodes text vectors.
    Returns the new data set of feature vectors
'''
def featurize(data, vocab):
    n = len(vocab)
    new_data = []
    for ex in data:
        vec = np.zeros(n + 1)
        for w in ex.split(" "):
            val = vocab[w] if w in vocab else -1
            vec[val + 1] += 1
        vec[0] = 1
        new_data.append(vec)
        
    return new_data

def load_data(filename, plusminus=False):

    data, labels = None, None
    with open(filename, 'r') as f:
        lines = f.readlines()
        labels = np.zeros(len(lines))
        data = [""] * labels.size
        print labels.size

        for i in xrange(len(lines)):
            line = lines[i]
            ind = line.find(" ")
            first, second = line[:ind], line[ind:]
            if plusminus:
                labels[i] = 1 if first == "+1" else -1 
            else:
                labels[i] = 1 if first == '+1' else 0
            data[i] = second

    return data, labels

def grade_output(y_hat, y):
    error = 0
    for i in range(y.size):
        if y_hat[i] != y[i]:
            error += 1
    print "error: ", error / (1.0 * y.size)

''' Loads the sentiment dataset and returns the 
    pair of (train_data, dev_data), which are themselves
    pairs of feature vectors x and labels y. 
'''  

def load_sentiment(pm=False):
    data, labels = load_data("polarity.train", plusminus=pm)
    vocab = vocabulary(data)
    vectors = featurize(data, vocab)
    train_data = [ (vectors[i], labels[i]) for i in range(len(labels)) ]

    d, l = load_data("polarity.dev", plusminus=pm)
    dev_vec = featurize(d, vocab)
    dev_data = [ (dev_vec[i], l[i]) for i in range(len(l))]
    return train_data, dev_data

''' Runs the model on the given data set '''
def run_sentiment(model, pm=False):
    train_data, dev_data = load_sentiment(pm)
    model.train(train_data)
    model.test(dev_data)

from nn import *
if __name__ == '__main__':
    model_a = LogisticClassifier(reg=0.001)
    model_b = HingeLossClassifier(reg=0.01)
    # Shallow neural network, h = 100
    model_c = NeuralNetwork(11700, 10, 1)
    run_sentiment(model_a)
    run_sentiment(model_b, True)
    #run_sentiment(model_c)
