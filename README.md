# dist-dqn
Distributed Reinforcement Learning using Deep Q-Network in TensorFlow.

Distributed DQN framework for training OpenAI Gym (https://gym.openai.com/) environments over multiple GPUs. Can also be configured to run in a cluster of hosts.

**Single node training:** `./scripts/dqn_single_node.sh <env_name>`  
**Multi-GPU training:** `./scripts/dqn_multi_gpu.sh <env_name> <num_gpus>`  

Implements a simple fully-connected network with two hidden layers for small environments like CartPole (https://gym.openai.com/envs/CartPole-v0) as well as the convolutional network architecture described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf for enviroments such as Pong (https://gym.openai.com/envs/Pong-v0).

**TODO: More info soon!**
