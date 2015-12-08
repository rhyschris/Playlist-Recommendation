import glm
import nn
import sys
import numpy as np
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
def load(filename):

	data = []
	with open( "{0}/{1}.txt".format(_datadir, filename), 'r') as f:
		linecount, n = 0, 0
		for line in f:
			if linecount == 0:
				n = int(line)
			else:
				split = [float(num) for num in line.split(' ')]
				x, y = np.array(split[:n]), int(split[-1])
				data.append((x, y))
			linecount += 1
	return data


''' Performs any averaging on feature sets'''
def preprocess(dataset):
	# Stub: do nothing yet
	return dataset

''' Entry point for loading and running a dataset.  '''
def main():
	train_set = preprocess( load('train'))
	test_set = preprocess( load('test'))

	model = glm.LogisticClassifier(epochs=200, reg=0, alph=1.0)
	model.train(train_set)
	model.test(train_set)	

if __name__ == '__main__':
	main()


