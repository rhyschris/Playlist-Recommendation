import numpy as np
import sys
import math
from sklearn import linear_model


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
            vec[ vocab[w] + 1 ] += 1
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
        
''' Binary logistic regression '''
def output(theta, x):
    return 1 if logistic(np.dot(theta, x)) > 0.5 else 0

''' The output '''
def df(y, theta, x):
    prod = logistic(np.dot(theta, x))
    return (prod - y) * x

def logistic(z):
    return 1.0 / (1.0 + np.exp(-1.0 * z))

def train(data, labels, num_feat):
    # Calculate objectives

    epochs = 20
    theta = np.zeros(num_feat + 1)
    for it in range(epochs):

        alpha = 0.01
        for i in range(len(labels)):
            x_i = data[i]
            y_i = labels[i]

            predict = output(theta, x_i)
            if predict != y_i:
                theta -= alpha * df(y_i, theta, x_i)

    return theta

def predict(data, theta):
    y = np.ones(len(data))
    
    for i in range(y.size):
        x_i = data[i]
        res = logistic(np.dot(theta, x_i))
        if res < 0.5:
            y[i] = 0 

    return y
        

''' Takes two numpy arrays '''
def test(x, y):
    diff = int(sum(abs(x - y)))
    print "Error: ", diff / (1.0 * y.shape[0])
    

if __name__ == "__main__":
    data, labels = load_data("polarity.train")
    vocab = vocabulary(data)
    vectors = featurize(data, vocab)
    theta = train(vectors, labels, len(vocab))
    y = predict(vectors, theta)

    test(labels, y)




