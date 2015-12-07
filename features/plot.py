from tsne import *
import numpy as np

''' Takes fft data and plots it'''
arr = []
with open('fftTransitions.txt', 'r') as f:
    # Take off the label
    lines = [l.rstrip() for l in f]
    n = int(lines[0])
    half = n / 2
    
    
    it = 0
    for l in lines[1:]:
        
        it += 1
        prev = np.array([ float(num) for num in l[:-2].split(" ")[:half]])

        next = np.array([ float(n) for n in l[:-2].split(" ")[half:]])
        prevsong = prev.reshape((96, 50))
        nextsong = next.reshape((96, 50))
        
        prevsong = np.sum(prevsong, axis=1)
        nextsong = np.sum(nextsong, axis=1)
        arr.append(np.array(list(prevsong) + list(nextsong)))
    

''' read in labels '''
with open('labels.txt', 'r') as l:
    labels = [word.rstrip() for word in l.readlines()]
    

matr = np.array( [ np.array(entry) for entry in arr] )
''' plot scatter '''

Y = tsne(matr)
import matplotlib.pyplot as plt

plt.scatter(Y[:, 0], Y[:, 1], 20)
for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(-10, 10), 
                 textcoords="offset points", 
                 bbox = dict(boxstyle='round', fc="yellow"))

plt.savefig("test.ps", format='eps', dpi=1000)
plt.show()


