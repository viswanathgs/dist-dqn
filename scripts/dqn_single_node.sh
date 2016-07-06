#!/bin/bash

# DQN on a single node with 0 or 1 GPU

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <env_name>"
  exit 1
fi

SCRIPTS_DIR=$( cd "$(dirname "${BASH_SOURCE}")" ; pwd -P )

source "$SCRIPTS_DIR/dqn_params.sh"

env_name=$1
if [[ "$env_name" == "cartpole" ]]; then
  DQN_PARAMS=$CARTPOLE
elif [[ "$env_name" == "pong" ]]; then
  DQN_PARAMS=$PONG
else
  echo "Invalid env_name"
  exit 1
fi

python "$SCRIPTS_DIR/../src/main.py" $DQN_PARAMS
