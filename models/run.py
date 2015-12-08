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
def load(filename, plusminus=False):

	data = []
	with open( "{0}/{1}.txt".format(_datadir, filename), 'r') as f:
		linecount, n = 0, 0
		for line in f:
			if linecount == 0:
				n = int(line)
			else:
				split = [float(num) for num in line.split(' ')]
				x, y = np.array(split[:n]), int(split[-1])
				if plusminus and y == 0:
					y = -1	
				data.append((x, y))
			linecount += 1
	return data


''' Performs any averaging on feature sets'''
def preprocess(dataset):
	s = []

	for x,y in dataset:
		n = x.size
		s.append((x, y))
	return s

''' Entry point for loading and running a dataset.  '''
def main():
	train_set = preprocess( load('train', False))
	test_set = preprocess( load('test', False))
	i, h, o, = 1000, 25, 1
	model = glm.LogisticClassifier(epochs=100, reg=0, alph=5.0)
	
	model.train(train_set)
	print "Train performance:"
	model.test(train_set)
	print "Test performance:"
	model.test(test_set)

	print i, h, o

if __name__ == '__main__':
	main()


