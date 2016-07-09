#!/bin/bash

# Single-host multi-GPU DQN
# (Data parallelism using between-graph replication)

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <env_type> <env_name> <num_gpus>. " \
       "Options for env_type: [classic, atari]."
  exit 1
fi

# Get DQN hyper-parameters for the environment
SCRIPTS_DIR=$( cd "$(dirname "${BASH_SOURCE}")" ; pwd -P )
source "$SCRIPTS_DIR/dqn_params.sh"
dqn_params=$(dqn_params_for_env $1 $2)

BASE_LOG_DIR="/tmp"
TRAIN_LOG_DIR="$BASE_LOG_DIR/train"
GYM_LOG_DIR="$BASE_LOG_DIR/gym"

NUM_PARAM_SERVERS=1
PARAM_SERVER_PORT=8080
WORKER_PORT=8090

# num_gpus == 0: 1 CPU worker
# num_gpus == 1: 1 GPU worker
# num_gpus > 1: num_gpus-1 GPU workers (Param servers allocate all the
#               parameters on GPU 0, and allocating the graph replica on the
#               same device makes it run out of memory. So only use the
#               remaining devices for parallelism).
num_gpus=$3
if [[ "$num_gpus" -le 1 ]]; then
  NUM_WORKERS=1
else
  NUM_WORKERS=$(($num_gpus - 1))
fi

ps_hosts=""
i=0
while [[ $i -lt $NUM_PARAM_SERVERS ]]
do
  port=$(($PARAM_SERVER_PORT + $i))
  if [ -z $ps_hosts ]; then
    ps_hosts="localhost:$port"
  else
    ps_hosts="$ps_hosts,localhost:$port"
  fi

  i=$(($i + 1))
done

worker_hosts=""
i=0
while [[ $i -lt $NUM_WORKERS ]]
do
  port=$(($WORKER_PORT + $i))
  if [ -z $worker_hosts ]; then
    worker_hosts="localhost:$port"
  else
    worker_hosts="$worker_hosts,localhost:$port"
  fi

  i=$(($i + 1))
done

# Start the param servers
i=0
while [[ $i -lt $NUM_PARAM_SERVERS ]]
do
  outfile="$BASE_LOG_DIR/ps$i"
  echo "Starting param server $i. Stdout: $outfile, train logs: $TRAIN_LOG_DIR."

  python "$SCRIPTS_DIR/../src/main.py" \
  --logdir=$TRAIN_LOG_DIR \
  --ps_hosts=$ps_hosts \
  --worker_hosts=$worker_hosts \
  --job="ps" \
  --task_id=$i \
  > $outfile 2>&1 &

  i=$(($i + 1))
done

# Start the worker instances for each GPU
i=0
while [[ $i -lt $NUM_WORKERS ]]
do
  outfile="$BASE_LOG_DIR/worker$i"
  gym_log="$GYM_LOG_DIR$i"
  log_params="--logdir=$TRAIN_LOG_DIR --monitor --monitor_path=$gym_log \
              --disable_video" # Disable video for headless machines.

  echo "Starting worker $i. Stdout: $outfile, train logs: $TRAIN_LOG_DIR, " \
       "Gym monitor logs: $gym_log."
  python "$SCRIPTS_DIR/../src/main.py" $dqn_params $log_params \
  --logdir=$TRAIN_LOG_DIR \
  --ps_hosts=$ps_hosts \
  --worker_hosts=$worker_hosts \
  --job="worker" \
  --task_id=$i \
  --gpu_id=$(($i + 1)) \
  > $outfile 2>&1 &

  i=$(($i + 1))
done
