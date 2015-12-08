# -*- coding: utf-8 -*-
import glm
import numpy as np
import scipy
import random
import sys
import copy

# Given up to num training examples, generates a boolean function.
def bool_fn(num):
    data = []
    labels = []
    for i in range(num):
        a = 0 if num % 2 else 1
        b = 0 if (num / 2) % 2 else 1
        c = 0 if (num / 4) % 2 else 1
        d = 0 if (num / 8) % 2 else 1
        data.append( np.array([a, b, c, d, 1.0])) 
        truth_value = (a == 1 and b == 1) ^ (c == 1 or d == 1)
        if truth_value:
            labels.append(0)
        else:
            labels.append(1)
    return data, np.array(labels) 
def circle_fn():
    data = [ [random.uniform(0, 10), random.uniform(0, 10), 1] for i in range(1000)]
    labels = [ 1 if elem[0]**2 + elem[1]**2 > 30 else 0 for elem in data]
    return [[ data[i], labels[i]] for i in range(len(labels))]

def test_bool():
    nn = NeuralNetwork(3, 5, 1)
    dataset = circle_fn()
    nn.train(dataset)
    nn.test(dataset)


from numpy import outer, dot

def randmatrix(m, n, lower=-0.5, upper=0.5):
    """Creates an m x n matrix of random values in [lower, upper]"""
    return np.array([random.uniform(lower, upper) for i in range(m*n)]).reshape(m, n)


class NeuralNetwork(glm.Model):
    def __init__(self, input_dim=0, hidden_dim=0, output_dim=0, afunc=np.tanh, d_afunc=(lambda z : 1.0 - z**2)):        
        self.afunc = afunc 
        self.d_afunc = d_afunc      
        self.input = np.ones(input_dim+1)   # +1 for the bias                                         
        self.hidden = np.ones(hidden_dim+1) # +1 for the bias        
        
        self.output = np.ones(output_dim)        
        self.iweights = randmatrix(input_dim+1, hidden_dim)
        self.inG = np.ones((input_dim +1)* hidden_dim).reshape((input_dim + 1, hidden_dim))
        self.oweights = randmatrix(hidden_dim+1, output_dim)        
        self.outG = np.ones((hidden_dim + 1) * output_dim).reshape((hidden_dim + 1, output_dim))

        self.oerr = np.zeros(output_dim)
        self.ierr = np.zeros(input_dim+1)
        self.i_size = self.iweights.size

    # Calculates the objective function (squared error) from the data. 
    # Utilizes forward_propagation to get the output through the network.
    def objective(self, matr):
        error = 0.0        
        # Extract the most recent unflattened weights 
        self.iweights, self.oweights = matr[ : self.i_size].reshape(self.iweights.shape), matr[self.i_size : ].reshape(self.oweights.shape) 
        for ex, labels in self.training_data:
            self.forward_propagation(ex)
            error += np.sum(0.5 * (labels - self.output) ** 2)
        return error
    
    # Gradient of the objective function.
    def gradf(self, matr):
        
        igrad, ograd = np.zeros(self.iweights.shape), np.zeros(self.oweights.shape)
        oweight = matr[ igrad.size :].reshape(self.oweights.shape)
        
        # Gradient updates batched over all train samples
        for ex, labels in self.training_data:
             self.forward_propagation(ex)
             labels = np.array(labels)
             self.oerr = (labels-self.output) * self.d_afunc(self.output)
             herr = dot(self.oerr, oweight.T) * self.d_afunc(self.hidden)
             ograd -= outer(self.hidden, self.oerr)
             igrad -= outer(self.input, herr[:-1])
        
        return concatMatrix(np.zeros(matr.shape), igrad, ograd, reshape=True)
        
    def forward_propagation(self, ex):        
        self.input[ : -1] = ex # ignore the bias
        self.hidden[ : -1] = self.afunc(dot(self.input, self.iweights)) # ignore the bias
        self.output = self.afunc(dot(self.hidden, self.oweights))
        return copy.deepcopy(self.output)
        
    def backward_propagation(self, labels, alpha=1.0):
        labels = np.array(labels)       
        self.oerr = (labels-self.output) * self.d_afunc(self.output)
        herr = dot(self.oerr, self.oweights.T) * self.d_afunc(self.hidden)
        ograd = outer(self.hidden, self.oerr)
        igrad = outer(self.input, herr[:-1])

        self.oweights += alpha * ograd / np.sqrt(self.outG)
        self.iweights += alpha * igrad/  np.sqrt(self.inG)
        self.inG += igrad * igrad
        self.outG += ograd * ograd
        return np.sum(0.5 * (labels-self.output)**2)
    
    # Trains a shallow neural network using BFGS.    
    def train_opt (self, training_data, maxiter=500, alpha = 0.05, epsilon = 1.5e-8, display_progress = False):
        x_init = np.zeros(self.iweights.size + self.oweights.size)
        self.training_data = training_data
        x_init = concatMatrix(x_init, self.iweights, self.oweights, reshape=True)

        res = scipy.optimize.minimize (fun=self.objective, x0 = x_init, method = 'L-BFGS-B', \
                                       jac=self.gradf, tol = epsilon, \
                                       options = {'disp':True, 'maxiter' : maxiter} )
        if res.success:
            iw = self.iweights.size
            self.iweights = res.x[ : iw].reshape(self.iweights.shape)
            self.oweights = res.x[iw : ].reshape(self.oweights.shape)  
        else:
            print res.message
            

    def train(self, training_data, epochs=20):
        
        iteration = 0
        while iteration < epochs:
            error = 0.0
            random.shuffle(training_data)
            attempt = 0
            for ex, labels in training_data:
                self.forward_propagation(ex)
                error += self.backward_propagation(labels, alpha=0.5)
            	attempt += 1
            if True:
                print 'completed iteration %s; error is %s' % (iteration, error)
            iteration += 1
                    
    def predictor(self, ex):
        out = self.forward_propagation(ex)
        return 1 if out > 0.0 else -1
    
    def hidden_representation(self, ex):
        self.forward_propagation(ex)
        return self.hidden

# Puts the INPUT and OUTPUT matrixes inside the larger sparse matrix, TARGET.                                                                                                                         
def concatMatrix(target, input, output, reshape=False):
    isz, osz = input.size, output.size
    if reshape:
        input, output = input.reshape(isz), output.reshape(osz) 
    target[ : isz] = input
    target[isz : ] = output
    return target

if __name__ == '__main__':
	test_bool()
