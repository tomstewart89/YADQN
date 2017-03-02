from policy import eGreedy
from value_function import deepQNetwork
from experience_replay import experienceReplay, memoryNode
import numpy as np

# The experience replay, policy and value function is all encapsulated up into this agent class. The agent
# select actions (enact the eGreedy policy), observe the consequences (store the S,A,R,S' info in experience replay)
# and then reflect on all of this (train its Q-function). 

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
    
            # if the agent moves to the terminal state then the return is exactly the reward
            if memory.next.S is None:
                targets[i,memory.A] = memory.R

            # otherwise we bootstrap the return by observing the current reward and adding it to the value of the next state-greedy action 
            else:
                targets[i,memory.A] = memory.R + self.gamma * np.max(self.Q.predict(memory.next.S[np.newaxis]))
            
            states[i] = memory.S
                  
        # in case the experience replay wasn't able to serve up enough memories, we need to trim the matrices                  
        states.resize((i+1,self.stateDim))
        targets.resize((i+1,len(self.actions)))

        # and finally we pass this to the Q function for fitting
        self.Q.fit(states, targets)
