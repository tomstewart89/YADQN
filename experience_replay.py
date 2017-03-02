from collections import deque
import random as rnd

# Deep Q-learning requires that the training data be randomly selected and fed into the network to avoid
# the so called catastrophic forgetting problem. The experienceReplay class does this by storing instances of 
# memoryNode in a big double ended queue and returning a random selection of them on calls to 'recall'

# The memoryNode class stores the state, action, reward info as well as pointing to the next node from that experience.
# The let's me get the next state but later on it'll help me implement an n-step return to try to improve my returns estimates

class memoryNode:
    def __init__(self, S = None, A = None, R = None, nextNode = None):
        self.S = S
        self.A = A
        self.R = R
        self.next = nextNode

    def __str__(self):
        return "S: {} A: {} R: {}" .format(self.S,self.A,self.R)
        
    def stepAhead(self, n):
        node = self
        while n > 0 and node.next != None:
            yield node
            n = n - 1
            node = node.next

class experienceReplay:
    
    def __init__(self,bufferSize):
        self.buffer = deque([],bufferSize)

    def recall(self,batchSize=32):
        j = 0
        idx = range(0,len(self.buffer)-1)
        rnd.shuffle(idx)
        
        # for every shuffled id
        for i in idx:
            
            # if the corresponding element has a nextState then yield it
            if self.buffer[i].next != None:
                yield self.buffer[i]
                j += 1
                
            # and if we've returned enough samples then break
            if j == batchSize:
                break
        
    def remember(self, state, action, reward, nextState):
        # if the buffer is empty or the last state was terminal then we need to start afresh
        if len(self.buffer) == 0 or self.buffer[-1].S == None:
            self.buffer.append(memoryNode(state))
            
        memory = memoryNode(nextState)
        
        # fill in the remaining details for the current state
        self.buffer[-1].A = action
        self.buffer[-1].R = reward
        self.buffer[-1].next = memory
        
        # append a new memory for the current state
        self.buffer.append(memory)
