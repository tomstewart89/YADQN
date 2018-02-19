# YADQN
YADQN is yet another implementation of Deep Q-Networks (DQN) described in the Deepmind paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602). 

In this repository, DQN is used to control the [CartPole-v1](https://gym.openai.com/envs/CartPole-v1/) environment on the openai gym. To do that, a Q-function is implemented in Keras to map a state into the relative value of applying a LEFT or a RIGHT force to the cart. The Q-function is then trained by minimising the TD error from a set of S,A,R,S' tuples randomly sampled from an experience replay buffer. Lastly, an e-greedy policy selects an action for a given state to control the cart.

## Results:
On most runs, the algorithm was able to learn to balance the cart after a few hundred episodes:

#### Still learning

![64](https://user-images.githubusercontent.com/2457362/36380100-6c987f76-15c4-11e8-9e60-6b849ab685d6.gif)

#### Getting there...

![343](https://user-images.githubusercontent.com/2457362/36380102-6cc4a880-15c4-11e8-9371-d8534bb96bfa.gif)

#### Nailed it!

![512](https://user-images.githubusercontent.com/2457362/36380103-6cf2f55a-15c4-11e8-84b7-4c05b39b6470.gif)
