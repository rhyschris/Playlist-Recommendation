import heapq
import math
import copy
''' Given a sequence x, determines the optimal ordering for a linear
    chain of x.
'''
def decode(x, transition):
    n = len(x)
    # Construct Transition matrix
    P = [ [ log(transition(x[i], x[j])) for j in range(n) ] for i in range(n)]
    
    pq = PriorityQueue()
    
    start = [ ([i, j], P[i][j]) for j in range(n) for i in range(n) ]
    for elem in start:
        pq.enqueue(elem)
    while not pq.empty():
        state = LinearChain(pq.dequeue(), n)
        for song in state.transitions():
            s_left, s_right = LinearChain(state, n), LinearChain(state, n)
            pass # TODO: FINISH
            
''' Linear chain '''
class LinearChain:
    def __init__(self, pair, n):
        self.n = n
        if pair:
            self.chain = list(pair[0])
            self.prob = pair[1]
        else:
            self.chain = []
            self.prob = float('inf')
    
    def front():
        return self.chain[0]
    def end():
        return self.chain[-1]
    def intern():
        return self.chain, self.prob
    def isDone(self):
        return len(self.chain) == n

    def addLeft(num, pr):
        self.chain.insert(0, num)
        self.prob = pr if self.prob == float('inf') else (self.prob + pr)
        return self

    def addRight(num, pr):
        self.chain.append(num)
        self.prob = pr if self.prob == float('inf') else (self.prob + pr)
        return self
    
    ''' Determine what states could be reached from here. '''
    def transitions():
        return [i for i in range(self.n) if i not in self.chain]
    

# Wrapper around heap queue.
class PriorityQueue:
    def  __init__(self):
        self.DONE = 1000000
        self.heap = []
        self.priorities = {}  # Map from state to priority                                                                                                                                                                                                                    
    def empty():
        return len(self.heap) == 0
        
    def enqueue(self, state, newPriority):
        oldPriority = self.priorities.get(state)
        if oldPriority == None or (-1. * newPriority) < oldPriority:
            self.priorities[state] = -1. * newPriority
            heapq.heappush(self.heap, (-1. * newPriority, state))
            return True
        return False

    def dequeue(self):
        while len(self.heap) > 0:
            priority, state = heapq.heappop(self.heap)
            if self.priorities[state] == self.DONE: continue  # Outdated priority, skip                                                                                                                                                                                       
            self.priorities[state] = self.DONE
            return (state, priority)
        return (None, None) # Nothing left... 
