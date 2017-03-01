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
    
    def __init__(self, greediness = 0.1, decay=0.0):
        self.greediness = greediness
        self.decay = decay
        
    def enact(self, actions, values):
        
        # assume it's the greedy policy to begin with
        action = actions[np.argmax(values)]
        
        # but if we happen to want to explore, return the other action
        if rnd.random() < self.greediness:
            action = np.random.choice(actions)
        
        # it's good to become more greedy as time goes by, this let's move from exploration 
        # to exploitation as the environment becomes more understood
        self.greediness -= self.greediness * self.decay
        
        return action

#%%

# represents the Q-function, it accepts a value target, and a state. It then computes the value for all actions and returns them 
class deepQNetwork:
    
    def __init__(self, learningRate, noOfStateVariables, noOfActions): # we'll return a prediction of the value of each action here, when we fit we just set the value of the current q's
        
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

class memoryTrace:
    def __init__(self, S = None, A = None, R = None, nextTrace = None):
        
        self.S = S
        self.A = A
        self.R = R
        self.nextTrace = nextTrace

    def __str__(self):
        return "S: {} A: {} R: {}" .format(self.S,self.A,self.R)
        
    def stepAhead(self, n):

        trace = self
        while n > 0 and trace.nextTrace != None:
            yield trace
            n = n - 1
            trace = trace.nextTrace

#%%

class experienceReplay:
    
    def __init__(self,bufferSize):
        
        self.buffer = deque([],bufferSize)

    def sample(self,noOfSamples=32):
        
        j = 0
        idx = range(0,len(self.buffer)-1)
        rnd.shuffle(idx)
        
        # for every shuffled id
        for i in idx:
            
            # if the corresponding element has a nextState then yield it
            if self.buffer[i].nextTrace != None:
                yield self.buffer[i]
                j += 1
                
            # and if we've returned enough samples then break
            if j == noOfSamples:
                break
        
    def append(self, state, action, reward, nextState):

        trace = memoryTrace(state, action, reward)
    
        # if the buffer is empty or the last state was terminal then we need to start afresh
        if len(self.buffer) == 0 or self.buffer[-1].S == None:
            self.buffer.append(memoryTrace(state))
            
        trace = memoryTrace(nextState)
        
        # fill in the remaining details for the current state
        self.buffer[-1].A = action
        self.buffer[-1].R = reward
        self.buffer[-1].nextTrace = trace
        
        # append a new trace for the current state
        self.buffer.append(trace)

#%% Now let's encapsulate all this in an agent class

class agent:
    
    def __init__(self, stateDim, actions, learningRate=0.01, gamma=0.99, greediness=0.5, greedDecay=0.0):
        
        self.gamma = gamma
        
        self.stateDim = stateDim
        self.actions = actions
        
        self.policy = eGreedy(greediness,greedDecay)
        self.Q = deepQNetwork(learningRate, stateDim, len(actions))
        self.experience = experienceReplay(10000)     
        
    def act(self,state):
        
        return self.policy.enact(self.actions, self.Q.predict(state[np.newaxis,:]))
        
    def observe(self, state, action, reward, nextState):
        
        self.experience.append(state, action, reward, nextState)

    # By which I mean run through some experience and update the Q function accordingly
    def reflect(self, samples=100):
        
        targets = np.zeros((samples,len(self.actions)))
        states = np.zeros((samples,self.stateDim))
                        
        for (i, trace) in enumerate(self.experience.sample(samples)):
    
            targets[i] = self.Q.predict(trace.S[np.newaxis])
    
            # if the agent moves to the terminal state the best it can hope for is a zero reward              
            if trace.nextTrace.S is None:
                targets[i,trace.A] = trace.R

            else:
                targets[i,trace.A] = trace.R + self.gamma * np.max(self.Q.predict(trace.nextTrace.S[np.newaxis]))
            
            states[i] = trace.S
                  
        # fit the Q function using some more experience
        self.Q.fit(states, targets)

#%%

LEFT = 0
RIGHT = 1

blob = agent(4,[LEFT,RIGHT],greediness=0.05)
env = gym.make('CartPole-v1')
env = wrappers.Monitor(env, '/tmp/cartpole-experiment-v1')

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
