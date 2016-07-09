#!/bin/bash

# DQN on a single node with 0 or 1 GPU

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <env_type> <env_name>. " \
       "Options for env_type: [classic, atari]."
  exit 1
fi

# Get DQN hyper-parameters for the environment
SCRIPTS_DIR=$( cd "$(dirname "${BASH_SOURCE}")" ; pwd -P )
source "$SCRIPTS_DIR/dqn_params.sh"
dqn_params=$(dqn_params_for_env $1 $2)

# Logging and Gym monitoring params
BASE_LOG_DIR="/tmp"
TRAIN_LOG_DIR="$BASE_LOG_DIR/train"
GYM_LOG_DIR="$BASE_LOG_DIR/gym"
log_params="--logdir=$TRAIN_LOG_DIR --monitor --monitor_path=$GYM_LOG_DIR \
            --disable_video" # Disable video for headless machines.

# Run the DQN on a single node
echo "Starting DQN. Train logs: $TRAIN_LOG_DIR, Gym monitor logs: $GYM_LOG_DIR"
python "$SCRIPTS_DIR/../src/main.py" $dqn_params $log_params
