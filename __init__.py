#%%

import gym
from gym import wrappers
from collections import deque
import numpy as np
import random as rnd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt

#%% 

class eGreedy:
    
    def __init__(self, eExplore = 0.1, eExploit = 0.05, decay=0.0):
        self.epsilon = eExplore
        self.eExploit = eExploit
        self.decay = decay
        
    def enact(self, actions, values):
        
        # assume it's the greedy policy to begin with
        action = actions[np.argmax(values)]
        
        # but if we happen to want to explore, return a random action
        if rnd.random() < self.epsilon:
            action = np.random.choice(actions)
        
        # this let's the policy move from exploration to exploitation as the agent better understands the environment
        self.epsilon -= (self.epsilon - self.eExploit) * self.decay
        
        return action

#%%

# represents the Q-function, it accepts a value target, and a state. It then computes the value for all actions and returns them 
class deepQNetwork:
    
    def __init__(self, learningRate, noOfStateVariables, noOfActions):
        self.model = Sequential()
        self.model.add(Dense(100, input_dim=noOfStateVariables, activation='relu'))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(noOfActions,activation='linear')) 
        self.model.compile(lr=learningRate, optimizer='rmsprop', loss='mse')

    def predict(self, states):        
        return self.model.predict(states,batch_size=1)
    
    def fit(self, states, targets):
        self.model.fit(states, targets, verbose=False)

#%%

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

#%%

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

#%% Now let's encapsulate all this in an agent class

class agent:
    
    def __init__(self, stateDim, actions, learningRate=0.01, gamma=0.99, epsilon=0.1, memorySize=10000):
        self.gamma = gamma
        self.stateDim = stateDim
        self.actions = actions
        self.policy = eGreedy(epsilon)
        self.Q = deepQNetwork(learningRate, stateDim, len(actions))
        self.experience = experienceReplay(memorySize)
        
    def act(self,state):
        return self.policy.enact(self.actions, self.Q.predict(state[np.newaxis,:]))
        
    def observe(self, state, action, reward, nextState):
        self.experience.remember(state, action, reward, nextState)

    # By which I mean run through some experience and update the Q function accordingly
    def reflect(self, batchSize=100):
        targets = np.zeros((batchSize,len(self.actions)))
        states = np.zeros((batchSize,self.stateDim))
                        
        for (i, memory) in enumerate(self.experience.recall(batchSize)):
    
            targets[i] = self.Q.predict(memory.S[np.newaxis])
    
            # if the agent moves to the terminal state the best it can hope for is a zero reward              
            if memory.next.S is None:
                targets[i,memory.A] = memory.R

            else:
                targets[i,memory.A] = memory.R + self.gamma * np.max(self.Q.predict(memory.next.S[np.newaxis]))
            
            states[i] = memory.S
                  
        # fit the Q function using some more experience
        self.Q.fit(states, targets)

#%%

LEFT = 0
RIGHT = 1

blob = agent(4,[LEFT,RIGHT], epsilon=0.05)
env = gym.make('CartPole-v1')
#env = wrappers.Monitor(env, '/tmp/cartpole-experiment-v1')

t = 0
epLen = deque([],100)

for i_episode in range(700):
    
    S = env.reset()
    done = False   
    t = 0
    
    while not done:
        
        t += 1
        A = blob.act(S)
        S_dash, R, done, info = env.step(A)
        
        blob.observe(np.copy(S),A,R,np.copy(S_dash))
        
        S = np.copy(S_dash)
        
    # every now and then stop, and think things through:
    blob.reflect()

    # when the episode ends the agent will have hit a terminal state so give it a zero reward
    if t < 500:
        blob.observe(np.copy(S),A,0.,None)
    else:
        blob.observe(np.copy(S),A,1.,None)
            
    epLen.append(t)
    
    print("episode: {}, average: {}".format(i_episode,np.mean(epLen)))

env.close()
#%%

gym.upload('/tmp/cartpole-experiment-v1', api_key='MY_API_KEY')
