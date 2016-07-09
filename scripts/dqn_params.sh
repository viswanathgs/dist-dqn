#!/bin/bash

# DQN params for various types of environments

# Hyper-parameters for classic control environments from RL literature.
# https://gym.openai.com/envs#atari.
CONTROL="\
--network=simple \
--optimizer=adam \
--lr=0.001 \
--minibatch_size=100 \
--num_episodes=10000 \
--max_steps_per_episode=200 \
--replay_memory_capacity=50000 \
--target_update_freq=3000 \
--reward_discount=0.9 \
--init_random_action_prob=0.5 \
--min_random_action_prob=0.1 \
--random_action_explore_steps=50000 \
"

# Hyper-parameters for Atari environments.
# https://gym.openai.com/envs#atari.
ATARI="\
--network=cnn \
--optimizer=rmsprop \
--lr=0.00025 \
--minibatch_size=32 \
--num_episodes=10000 \
--max_steps_per_episode=500000 \
--replay_memory_capacity=1000000 \
--target_update_freq=10000 \
--reward_discount=0.99 \
--init_random_action_prob=1.0 \
--min_random_action_prob=0.1 \
--random_action_explore_steps=1000000 \
--frames_per_state=4 \
--update_freq=4 \
--replay_start_size=10000 \
--resize_width=84 \
--resize_height=84 \
"

dqn_params_for_env() {
  if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <env_type> <env_name>. " \
         "Options for env_type: [classic, atari]."
    exit 1
  fi

  env_type=$1
  env_name=$2

  if [[ "$env_type" == "control" ]]; then
    params=$CONTROL
  elif [[ "$env_type" == "atari" ]]; then
    params=$ATARI
  else
    echo "Invalid env_type $env_type. Choices are [control, atari]."
    exit 1
  fi

  echo "--env=$env_name $params"
}
