#!/usr/local/bin/python3

import argparse
import gym
import logging
import tensorflow as tf

from dqn_agent import DQNAgent
from network import Network
from replay_memory import ReplayMemory

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--log_level', default='INFO', help='Log verbosity')
  
  # Environment
  parser.add_argument('--env', default='CartPole-v0', 
    help='OpenAI Gym environment name')
  parser.add_argument('--monitor', action='store_true',
    help='Turn on OpenAI Gym monitor')
  parser.add_argument('--monitor_path', default='/tmp/gym',
    help='Path for OpenAI Gym monitor logs')
  
  # Network
  parser.add_argument('--network', default='simple', choices=['simple', 'cnn'],
    help='Network architecture type')
  parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
  parser.add_argument('--reg_param', default=0.001, type=float,
    help='Regularization param')
  parser.add_argument('--optimizer', default='sgd',
    choices=['adadelta', 'adagrad', 'adam', 'ftrl', 'sgd', 'momentum', 'rmsprop'],
    help='Type of optimizer for gradient descent')
  parser.add_argument('--momentum', default=0.9, type=float,
    help='Momentum value for MomentumOptimizer')
  parser.add_argument('--rmsprop_decay', default=0.9, type=float,
    help='Decay for RMSPropOptimizer')

  # Agent
  parser.add_argument('--num_episodes', default=10000, type=int,
    help='Number of episodes to train')
  parser.add_argument('--max_steps_per_episode', default=1000, type=int,
    help='Max steps to train for each episode')
  parser.add_argument('--minibatch_size', default=30, type=int,
    help='Minibatch size for each training step')
  parser.add_argument('--frames_per_state', default=1, type=int,
    help='Number of consecutive frames that form a state')
  parser.add_argument('--resize_width', default=80, type=int,
    help='Resized screen width for frame pre-processing')
  parser.add_argument('--resize_height', default=80, type=int,
    help='Resized screen height for frame pre-processing')
  parser.add_argument('--reward_discount', default=0.9, type=float,
    help='Discount factor for future rewards')
  parser.add_argument('--replay_memory_capacity', default=10000, type=int,
    help='Max size of the memory for experience replay')
  parser.add_argument('--init_random_action_prob', default=0.5, type=float,
    help='Initial probability for choosing random actions')
  parser.add_argument('--random_action_prob_decay', default=0.99, type=float,
    help='Decay rate for random action probability')
  parser.add_argument('--min_random_action_prob', default=0.1, type=float,
    help='Threshold at which to stop decaying random action probability')
  parser.add_argument('--target_update_freq', default=300, type=int,
    help='Frequency for updating target network in terms of number of episodes')

  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()

  logging.getLogger().setLevel(args.log_level)

  env = gym.make(args.env)
  network = Network.create_network(
    input_shape=env.observation_space.shape,
    num_actions=env.action_space.n,
    config=args,
  )
  replay_memory = ReplayMemory(args.replay_memory_capacity)

  session = tf.Session()
  session.run(tf.initialize_all_variables())
  dqn_agent = DQNAgent(env, network, session, replay_memory, args)

  if args.monitor:
    env.monitor.start(args.monitor_path, force=True)
  dqn_agent.train(args.num_episodes, args.max_steps_per_episode)
  if args.monitor:
    env.monitor.close()
