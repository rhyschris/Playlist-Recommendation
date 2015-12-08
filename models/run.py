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
				if y == 0:
					y = -1	
				data.append((x, y))
			linecount += 1
	return data


''' Performs any averaging on feature sets'''
def preprocess(dataset):
	# Apply subsets of features here.
	processed = []
	for x, y in dataset:
		n = x.size
		processed.append( (np.array(list(x)[ int(n*0.4): int(n*0.6) ] ), y))
	return processed

''' Entry point for loading and running a dataset.  '''
def main():
	train_set = preprocess( load('train', True))
	test_set = preprocess( load('test', True))

	model = glm.LeastSquaresClassifier()

	model.train(train_set)
	print "Train performance:"
	model.test(train_set)
	print "Test performance:"
	model.test(test_set)

if __name__ == '__main__':
	main()


