from collections import deque
import gym
from gym import wrappers
import numpy as np
from agent import agent

LEFT = 0
STOP = 1
RIGHT = 2

MAX_TIMESTEPS = 200

blob = agent(2,[LEFT,STOP,RIGHT], epsilon=0.5)
env = gym.make('MountainCar-v0')

#env = wrappers.Monitor(env, '/tmp/cartpole-experiment-v1')

t = 0
episodeLength = deque([],100)

for i_episode in range(10000):
    
    S = env.reset()
    done = False   
    t = 0
    
    while not done:
        
        t += 1
        A = blob.act(S)
        S_dash, R, done, info = env.step(A)
        
        blob.observe(S,A,R,S_dash)
        
        S = np.copy(S_dash)
        
    # every now and then stop, and think things through:
    blob.reflect()
        
    # when the episode ends the agent will have hit a terminal state so give it a zero reward
    if t < MAX_TIMESTEPS:
        blob.observe(S,A,1.,None)
    else:
        blob.observe(S,A,0.,None)
            
    episodeLength.append(t)
    
    print("episode: {}, average: {}".format(i_episode,np.mean(episodeLength)))

env.close() 

#gym.upload('/tmp/cartpole-experiment-v1', api_key='MY_API_KEY')