#!/bin/bash

# DQN params for various environments

CARTPOLE="\
--env=CartPole-v0 \
--network=simple \
--optimizer=sgd \
--lr=0.001 \
--minibatch_size=30 \
--num_episodes=3000 \
--max_steps_per_episode=200 \
--replay_memory_capacity=10000 \
--target_update_freq=300 \
--reward_discount=0.9 \
--init_random_action_prob=0.9 \
--min_random_action_prob=0.1 \
--random_action_prob_decay=0.99 \
"

PONG="\
--env=Pong-v0 \
--network=cnn \
--optimizer=adam \
--lr=0.00001 \
--minibatch_size=100 \
--num_episodes=100000 \
--max_steps_per_episode=5000 \
--replay_memory_capacity=100000 \
--target_update_freq=300 \
--reward_discount=0.9 \
--init_random_action_prob=0.5 \
--min_random_action_prob=0.1 \
--random_action_prob_decay=0.99 \
--frames_per_state=4 \
--resize_width=80 \
--resize_height=80 \
"
