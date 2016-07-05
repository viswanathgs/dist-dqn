#!/bin/bash

# Single-host multi-GPU DQN 
# (Data parallelism using between-graph replication)

source "dqn_params.sh"

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <env_name> <num_gpus>"
  exit 1 
fi

env_name=$1
if [[ "$env_name" == "cartpole" ]]; then
  DQN_PARAMS=$CARTPOLE
elif [[ "$env_name" == "pong" ]]; then
  DQN_PARAMS=$PONG
else
  echo "Invalid env_name"
  exit 1
fi

# num_gpus == 0: 1 CPU worker
# num_gpus == 1: 1 GPU worker
# num_gpus > 1: num_gpus-1 GPU workers (Param servers allocate all the 
#               parameters on GPU 0, and allocating the graph replica on the
#               same device makes it run out of memory. So only use the remaining
#               devices for parallelism).
num_gpus=$2
if [[ "$num_gpus" -le 1 ]]; then
  NUM_WORKERS=1
else
  NUM_WORKERS=$(($num_gpus - 1))
fi

NUM_PARAM_SERVERS=1
PARAM_SERVER_PORT=8080
WORKER_PORT=8090

PS_HOSTS=""
i=0
while [[ $i -lt $NUM_PARAM_SERVERS ]]
do
  port=$(($PARAM_SERVER_PORT + $i))
  if [ -z $PS_HOSTS ]; then
    PS_HOSTS="localhost:$port"
  else
    PS_HOSTS="$PS_HOSTS,localhost:$port"
  fi

  i=$(($i + 1))
done

WORKER_HOSTS=""
i=0
while [[ $i -lt $NUM_WORKERS ]]
do
  port=$(($WORKER_PORT + $i))
  if [ -z $WORKER_HOSTS ]; then
    WORKER_HOSTS="localhost:$port"
  else
    WORKER_HOSTS="$WORKER_HOSTS,localhost:$port"
  fi

  i=$(($i + 1))
done

# Start the param servers
i=0
while [[ $i -lt $NUM_PARAM_SERVERS ]]
do
  outfile="/tmp/ps$i"
  echo "Starting param server $i, redirecting stdout to $outfile"

  python ./src/main.py \
  --ps_hosts=$PS_HOSTS \
  --worker_hosts=$WORKER_HOSTS \
  --job="ps" \
  --task_id=$i \
  > $outfile 2>&1 &

  i=$(($i + 1))
done

# Start the worker instances for each GPU
i=0
while [[ $i -lt $NUM_WORKERS ]]
do
  outfile="/tmp/worker$i"
  echo "Starting worker $i, redirecting stdout to $outfile"

  python ./src/main.py \
  --ps_hosts=$PS_HOSTS \
  --worker_hosts=$WORKER_HOSTS \
  --job="worker" \
  --task_id=$i \
  --gpu_id=$(($i + 1)) \
  $DQN_PARAMS \
  > $outfile 2>&1 &

  i=$(($i + 1))
done
