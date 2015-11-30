import numpy as np
import sys
import math
import copy
import random
from sklearn import linear_model
from scipy.special import expit
import scipy

# A classifier is something with a train and predict method.
class Classifier(object):
    
    def train(self, x, y, num_feat, epochs=None):
        # Calculate objectives
        if self.epochs is None:
            self.epochs = epochs if epochs else 20
        if not hasattr(self, 'theta'):
            self.theta = np.zeros(num_feat + 1)
        print self.theta

        for it in range(self.epochs):        
            alph = self.alpha(it)
            error = 0
            for i in range(len(y)):
                if self.predictor(x[i]) != y[i]:
                    error += 1
                if self.shouldUpdate(x[i], y[i]):
                    self.theta -= alph * self.df_theta(y[i], x[i])
            print "Iter {0} Error: {1}".format(it, error / (1.0 * len(y)))
        return self.theta
    ''' Runs the predictor on the test set and returns the error rate. '''

    def alpha(self, iter = 1):        
        return 1.0

    def test(self, x, y):
        y_hat = np.zeros(len(y))
        
        error = 0
        for i in range(y.size):
            y_hat[i] = self.predictor(x[i])
            if y_hat[i] != y[i]:
                error += 1

        print error / (1.0 * y.size)
        return error / (1.0 * y.size)

    ''' Runs the class predictor and outputs a label.  Override this in an subclass.'''
    def predictor(self, x_i):
        pass
    
    ''' Implements the standard online predictor logic '''
    def shouldUpdate(self, x_i, y_i):
        return True

    ''' Computes the gradient of the loss with respect to theta. '''
    def df_theta(self, y_i, x_i):
        pass

''' 
    Wrapper around an adagrad learner.  
    Uses the gradient history to minimize regret.
'''

class AdagradClassifier(Classifier):
    
    ''' Uses Adagrad to optimize a stochastic gradient descent operation
        efficiently. If full = True, does the full matrix regret minimization 
        procedure, or automatically if the feature space is very small.  Otherwise,
        uses the diagonal regret minimization procedure.
    ''' 
    def init_Gt(self):
        self.full = False
        G_t =  np.ones(self.num_params())
        return G_t

    def num_params():
        pass
    
    def train(self, x, y, num_feat, epochs=None, full=False):
        # Calculate objectives                                                                                                  
        if self.epochs is None:
            self.epochs = epochs if epochs else 20
            
        # Introspection, so that we don't redefine theta if it doesn't exist
        if not hasattr(self, 'theta'):
            self.theta = np.zeros(num_feat + 1)
        self.full = full
        eta = self.alpha(1)
        # Use full matrix history if your feature space is small enough.
        G_t = self.init_Gt()
        
        numex = len(y)
        for it in range(self.epochs):
            error = 0
            for i in range(numex):
                if y[i] != self.predictor(x[i]) :
                    error += 1
                if self.shouldUpdate(x[i], y[i]):
                    g_t = self.df_theta(y[i], x[i])
        
                    if self.full:
                        update = eta * np.dot(1.0 / np.sqrt(G_t), g_t)
                        G_t += np.outer(g_t, g_t)
                    else:
                        update = eta * g_t / G_t
                        G_t += (g_t * g_t)
                    self.theta -= update 
        
            print "Iter {0} Error: {1}".format(it, error / (1.0 * numex))        
           
        return self.theta

class LogisticClassifier(AdagradClassifier):
    def __init__(self, num_feat, epochs=None):
        self.epochs = epochs if epochs else 20
        self.alpha = lambda iter: 0.06
        self.num_params = lambda: num_feat + 1

    def predictor(self, x_i):
        return 1 if self.logistic(np.dot(self.theta, x_i)) > 0.5 else 0
    
    def df_theta(self, y_i, x_i):
        prod = self.logistic(np.dot(self.theta, x_i))
        return (prod - y_i) * x_i
    
    def logistic(self, z):
        return expit(z) 
       
class LeastSquaresClassifier(AdagradClassifier):
    def __init__(self, epochs=None, reg=0.001):
        self.epochs = epochs if epochs else 50
        self.alpha = lambda iter: 0.06
        self.reg = reg
    
    def predictor(self, x_i):
        return 1 if np.dot(self.theta, x_i) > 0.0 else 0
    
    def df_theta(self, y_i, x_i):
        return (np.dot(self.theta, x_i) - y_i) * x_i 

''' Implements a shallow neural network on a binary objective '''
class NeuralNetwork(AdagradClassifier):
    def __init__(self, epochs=None, reg = 0.001, h = 100, num_feat=10000):
        self.epochs = epochs if epochs else 5
        self.alpha = lambda iter: 1.0
        self.reg = reg
        self.h = h

        w_init = 2.0 * np.sqrt(6) / (np.sqrt(num_feat + h + 2))
        u_init = 2.0 * np.sqrt(6) / (np.sqrt(h + 4))
        self.W = w_init * (np.random.rand(h + 1, num_feat + 1) - 0.5)
        self.U = u_init * (np.random.rand(1, h + 1) - 0.5)
        print self.W.shape
        print self.U.shape

        self.theta = np.empty_like(self.W)
        np.copyto(self.theta, self.W)
        self.theta = self.append(self.U, self.theta.reshape((self.theta.size, )))
                
    # Number of parameters for "train"
    def num_params(self):
        return self.W.size + self.U.size 

    ''' Flattens the given matrix and appends it to the end of the destination vector. '''
    def append(self, matr, dst_vec):
        m_copy = np.empty_like(matr)
        np.copyto(m_copy, matr)
        flattened = np.reshape(m_copy, (m_copy.size, ))
        dst_vec = np.append(dst_vec, flattened)
        return dst_vec
        
    def unroll(self):
        self.W = self.theta[ : self.W.size].reshape(self.W.shape)
        self.U = self.theta[self.W.size:].reshape(self.U.shape)
        
    # Gives the feedfoward portion of the network
    def predictor(self, x_i):
       # print 'iinnnnn'
        self.unroll()
        self.z = np.tanh( np.dot(self.W, x_i))
        self.a = self.logistic(np.dot(self.U, self.z))
        y_hat = 1 if self.a >= 0.5 else 0
        return y_hat

    ''' Gives the backpropagation algorithm '''
    def df_theta(self, y_i, x_i):
        
        # Forward propagation - ignore the value, because we
        # always update the weights
        self.unroll()
        self.predictor(x_i)
        oerr = y_i - self.a
        U_grad = np.outer(oerr, self.z.T)
        # activation layer error
        d_activ = 1 - self.z*self.z
        ierr = np.dot(self.U.T, oerr) * d_activ
        W_grad = np.outer(ierr, x_i)
        d_theta = np.empty_like(W_grad)
        np.copyto(d_theta, W_grad)
        d_theta = self.append(U_grad, d_theta)
  
        return d_theta
    
    def logistic(self, z):
        return expit(z)
        
''' Hinge loss with regularization.'''
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
        
    def df_theta(self, y_i, x_i):
        dotprod = np.dot(self.theta, x_i)
        if dotprod * y_i < 1:
            return -1.0 * x_i * y_i + self.reg * self.theta
        else:
            return self.reg * self.theta
        
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
    nn = ShallowNeuralNetwork(3, 5, 1)
    dataset = circle_fn()
    nn.train(dataset)
    nn.test(dataset)


from numpy import outer, dot

def randmatrix(m, n, lower=-0.5, upper=0.5):
    """Creates an m x n matrix of random values in [lower, upper]"""
    return np.array([random.uniform(lower, upper) for i in range(m*n)]).reshape(m, n)


class ShallowNeuralNetwork:
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

        self.oweights += alpha  / np.sqrt(self.outG) * ograd
        self.iweights += alpha /  np.sqrt(self.inG) * igrad
        self.inG += igrad * igrad
        self.outG += ograd * ograd
        return np.sum(0.5 * (labels-self.output)**2)
    
    # Trains a shallow neural network using BFGS.    
    def train_opt (self, training_data, maxiter=50, alpha = 0.05, epsilon = 1.5e-8, display_progress = False):
        x_init = np.zeros(self.iweights.size + self.oweights.size)
        self.training_data = training_data
        x_init = concatMatrix(x_init, self.iweights, self.oweights, reshape=True)

        res = scipy.optimize.minimize (fun=self.objective, x0 = x_init, method = 'BFGS', \
                                           jac=self.gradf, tol = epsilon, options = {'disp':True, 'maxiter' : maxiter} )
        if res.success:
            iw = self.iweights.size
            self.iweights = res.x[ : iw].reshape(self.iweights.shape)
            self.oweights = res.x[iw : ].reshape(self.oweights.shape)  
        else:
            print res.message
            

    def train(self, training_data, maxiter=100, alpha=0.05, epsilon=1.5e-8, display_progress=True):       
        iteration = 0
        error = sys.float_info.max

        while error > epsilon and iteration < maxiter:    
            error = 0.0
            random.shuffle(training_data)
            for ex, labels in training_data:
                self.forward_propagation(ex)
                error += self.backward_propagation(labels, alpha=alpha)           
            if display_progress:
                print 'completed iteration %s; error is %s' % (iteration, error)
            iteration += 1
                    
    def predict(self, ex):
        self.forward_propagation(ex)
        return 1 if self.output[0] > 0.5 else 0
    def test(self, dataset):
        error = 0    
        for ex, label in dataset:
            if label != self.predict(ex):
                error += 1
        print "error: {0}".format(error / (1.0 * len(dataset)))

    
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

def run(reg=0.001):
    data, labels = load_data("polarity.train", plusminus=False)
    vocab = vocabulary(data)
    vectors = featurize(data, vocab)
    
    logit = LogisticClassifier(2)
    logit.train(vectors, labels, 2)
    
    d, l = load_data("polarity.dev", plusminus=False)
    dev_vec = featurize(d, vocab)
    logit.test(vectors, labels)


if __name__ == '__main__':
    test_bool()
    run()
