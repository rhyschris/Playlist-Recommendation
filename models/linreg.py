import numpy as np
import sys
import math
from sklearn import linear_model

# A classifier is something with a train and predict method.
class Classifier():
    
    def train(self, x, y, num_feat, epochs=None):
        # Calculate objectives
        if self.epochs is None:
            self.epochs = epochs if epochs else 20

        self.theta = np.zeros(num_feat + 1)
        for it in range(self.epochs):
            alph = self.alpha()
            for i in range(len(y)):
                predict = self.predictor(x[i])
                if predict != y[i]:
                    self.theta -= alph * self.df_theta(y[i], x[i])
        return self.theta
    ''' Runs the predictor on the test set and returns the error rate. '''

    def alpha(iter = 1):
        return 0.01

    def test(self, x, y):
        y_hat = np.zeros(len(data))
    
        error = 0
        for i in range(y.size):
            y_hat[i] = self.predictor(x[i])
            if y_hat[i] != y[i]:
                error += 1

        print error / (1.0 * y.size)
        return error / (1.0 * y.size)

    ''' Runs the class predictor.  Outputs a y label '''
    def predictor(self, x_i):
        pass
    ''' Computes the gradient of the loss with respect to theta. '''
    def df_theta(self, y_i, x_i):
        pass

class LogisticClassifier(Classifier):
    def __init__(self, epochs=None):
        self.epochs = epochs if epochs else 20

    def predictor(self, x_i):
        return 1 if self.logistic(np.dot(self.theta, x_i)) > 0.5 else 0

    def df_theta(self, y_i, x_i):
        prod = self.logistic(np.dot(self.theta, x_i))
        return (prod - y_i) * x_i

    def logistic(self, z):
        return 1.0 / (1.0 + np.exp(-1.0 * z))

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

def load_data(filename):

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
            labels[i] = 1 if first == "+1" else 0 
            data[i] = second

    return data, labels

if __name__ == "__main__":
    data, labels = load_data("polarity.train")
    vocab = vocabulary(data)
    vectors = featurize(data, vocab)


    reg = LogisticClassifier();
    reg.train(vectors, labels, len(vocab))
    print "train"
    reg.test(vectors, labels)

    print "dev"
    d, l = load_data("polarity.dev")
    dev_vec = featurize(d, vocab)

    reg.test(dev_vec, l)
