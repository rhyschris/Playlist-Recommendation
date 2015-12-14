import glm
import nn
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import *
from sklearn.ensemble import *
import sklearn.metrics as metrics
from sklearn.feature_selection import chi2

''' 
	Runs a classifier pipeline on the playlist data set. 
'''

_datadir = "../data/finished"


''' Loads datasets with the given name.
	Data sets are expected to be in the following format:

	(1) < size of feature vector>
	(2) x_i y_i
	.... 

'''
def load(filename, plusminus=False):

	data = []
	with open( "{0}/{1}.txt".format(_datadir, filename), 'r') as f:
		linecount, n = 0, 0
		for line in f:
			if linecount == 0:
				n = int(line)
			else:
				split = [float(num) for num in line.split(' ')]
				x_tentative = split[:n]
				x, y = np.array(x_tentative), int(split[-1])
				if plusminus and y == 0:
					y = -1	
				data.append((x, y))
			linecount += 1
	return data


''' Performs any averaging on feature sets'''
def preprocess(dataset, nn=False):
	s = []
	for x,y in dataset:
		n = x.size
		if nn:
			y_vec = [0, 0]
			y_vec[y] = 1
		else:
			y_vec = y
		s.append((x, y_vec))
	return s

''' Entry point for loading and running a dataset.  '''
def main(plot=False, cv=False):

	train_set = preprocess( load('train', True), nn=False)
	test_set = preprocess( load('test', True), nn=False)
	
	i, h, o, = 5000, 25, 2

	# Optimal models

	x_train = [x for x, y in train_set]
	y_train = [y for x, y in train_set]


	x_test = [x for x, y in test_set]
	y_test = [y for x, y in test_set]

	n = x_train[0].size
	scores, p_values = chi2(x_train, y_train) 

	significant = sum(1 for val in p_values if val < 0.7)
	print "Number of significant features: ", significant
	#model = glm.LogisticClassifier(epochs=50, reg=0, alph=10.0)
	#model = nn.NeuralNetwork(i, h, o, reg=1e-2)
	#model = glm.HingeLossClassifier(alpha=(lambda iter: 0.4 ), reg=0, epochs=50)
	
	# sklearn shieeeet
	#model = LogisticRegression(penalty='l1', solver='lbfgs')	
	#model = RandomForestClassifier(n_estimators=1000)


	# x_train = [x for x, y in train_set]
	# y_train = [y for x, y in train_set]


	# x_test = [x for x, y in test_set]
	# y_test = [y for x, y in test_set]


	# model.fit(x_train, y_train)
	# y_hat = model.predict(x_test)

	# print metrics.classification_report(y_test, y_hat)


	#model.cv(train_set, k=43)
	# model.train(train_set)
	# print "Train performance:"
	# model.test(train_set)
	# print "Test performance:"
	# model.test(test_set)


	if plot:
		notes = ["A", "Bflat", "B", "C", "C#", "D", "Eflat", "E", "F", "F#", "G", "Aflat"]

		# x = model.theta
		# x = x.reshape(96, 200)
		# print x.shape
		# y = np.sum(x / 200, axis=1)
		# y_norm = np.log(y)
		# r = range(1, 97)

		# print y_norm.shape
		# print ' '.join(str(yn) for yn in y_norm)

		plt.scatter(xrange(n), p_values)


		plt.xlabel("Feature place in window model")
		plt.ylabel("P-values")
		plt.title("Chi^2 MFCC Feature Significance")

		# labels = ["{0}{1}".format( notes[i%12], i/12) for i in r]

		# plt.xlabel = "Musical notes"
		# plt.ylabel = "Feature weights (Log)" 
		plt.show()

if __name__ == '__main__':
	main(True)


