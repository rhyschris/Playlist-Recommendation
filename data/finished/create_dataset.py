import random
import sys

# Makes the dataset from the two input labels
def make_data(trainf, testf):
    negf = open('label-0', 'r')
    posf = open('label-1', 'r')

    neglines = [ line for line in negf if line != '10400\n' ] 
    poslines = [ line for line in posf if line != '10400\n' ] 

    dataset = poslines + neglines
    for _ in range(3):
        random.shuffle(dataset)
    
    print len(dataset)
    thresh = int(0.6 * len(dataset))
    print thresh
    
    train, test = dataset[ : thresh], dataset[thresh : ]
    
    with open(trainf, 'w+') as f:
        for line in train:
            f.write(line)
    
    with open(testf, 'w+') as f:
        for line in test:
            f.write(line)
    
    print "DONE"
# ------------------------
if __name__ == '__main__':
    trainf = sys.argv[1]
    testf = sys.argv[2]
    
    make_data(trainf, testf)
