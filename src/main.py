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
  parser.add_argument('--disable_video', action='store_true',
    help='Disable video recording while when running the monitor')
  
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
  parser.add_argument('--init_random_action_prob', default=0.9, type=float,
    help='Initial probability for choosing random actions')
  parser.add_argument('--min_random_action_prob', default=0.1, type=float,
    help='Threshold at which to stop decaying random action probability')
  parser.add_argument('--random_action_explore_steps', default=10000, type=int,
    help='Number of steps over which to decay the random action probability')
  parser.add_argument('--target_update_freq', default=300, type=int,
    help='Frequency for updating target network in terms of number of episodes')

  # Distribution
  parser.add_argument('--ps_hosts', default='',
    help='Parameter Servers. Comma separated list of host:port pairs')
  parser.add_argument('--worker_hosts', default='localhost:0',
    help='Worker hosts. Comma separated list of host:port pairs')
  parser.add_argument('--job', default='worker', choices=['ps', 'worker'],
    help='Whether this instance is a param server or a worker')
  parser.add_argument('--task_id', default=0, type=int,
    help='Index of this task within the job')
  parser.add_argument('--gpu_id', default=0, type=int,
    help='Index of the GPU to run the training on')
  parser.add_argument('--sync', action='store_true',
    help='Whether to perform synchronous training')
  parser.add_argument('--disable_cpu_param_pinning', action='store_true',
    help='If set, param server will not pin the varaibles to CPU, allowing '
         'TensorFlow to default to GPU:0 if the host has GPU devices')
  parser.add_argument('--disable_target_replication', action='store_true',
    help='Unless set, the target params will be replicated on each GPU. '
         'Setting the flag defaults to a single set of target params managed '
         'by the param server.')

  # Summary
  parser.add_argument('--logdir', default='/tmp/train_logs',
    help='Directory for training summary and logs')
  parser.add_argument('--summary_freq', default=100, type=int,
    help='Frequency for writing summary (in terms of global steps')

  return parser.parse_args()

def run_worker(cluster, server, args):
  env = gym.make(args.env)
  worker_job = args.job

  # Have param server pin params to CPU unless specified otherwise.
  # If disabled and if the host has GPU support, /gpu:0 is used by default.
  ps_device = None if args.disable_cpu_param_pinning else '/cpu'

  # If no GPU devices are found, then allow_soft_placement in the
  # config below results in falling back to CPU.
  worker_device = '/job:%s/task:%d/gpu:%d' % \
                      (worker_job, args.task_id, args.gpu_id)

  replica_setter = tf.train.replica_device_setter(
    worker_device=worker_device,
    cluster=cluster,
  )
  with tf.device(replica_setter):
    network = Network.create_network(
      config=args,
      input_shape=DQNAgent.get_input_shape(env, args),
      num_actions=env.action_space.n,
      num_replicas=len(cluster.job_tasks(worker_job)),
      ps_device=ps_device,
      worker_device=worker_device,
    )
    init_op = tf.initialize_all_variables()

  # Designate the first worker task as the chief
  is_chief = (args.task_id == 0)

  # Create a Supervisor that oversees training and co-ordination of workers
  sv = tf.train.Supervisor(
    is_chief=is_chief,
    logdir=args.logdir,
    init_op=init_op,
    global_step=network.global_step,
    summary_op=None, # Explicitly disable as DQNAgent handles summaries
    recovery_wait_secs=5,
  )

  # Start the gym monitor if needed
  video = False if args.disable_video else None
  if args.monitor:
    env.monitor.start(args.monitor_path, force=True, video_callable=video)

  # Initialize memory for experience replay
  replay_memory = ReplayMemory(args.replay_memory_capacity)

  # Start the session and kick-off the train loop
  config=tf.ConfigProto(
    log_device_placement=True,
    allow_soft_placement=True,
  )
  with sv.managed_session(server.target, config=config) as session:
    dqn_agent = DQNAgent(
      env, network, session, replay_memory, args,
      enable_summary=is_chief, # Log summaries only from the chief worker
    )
    dqn_agent.train(args.num_episodes, args.max_steps_per_episode, sv)

  # Close the gym monitor
  if args.monitor:
    env.monitor.close()

  # Stop all other services
  sv.stop()

if __name__ == '__main__':
  args = parse_args()

  logging.getLogger().setLevel(args.log_level)

  ps_hosts = args.ps_hosts.split(',') if args.ps_hosts else []
  worker_hosts = args.worker_hosts.split(',') if args.worker_hosts else []
  cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})

  # Create a TensorFlow server that acts either as a param server or
  # as a worker. For non-distributed setup, we still create a single
  # instance cluster without any --ps_hosts and one item in --worker_hosts
  # that corresponds to localhost.
  server = tf.train.Server(
    cluster,
    job_name=args.job,
    task_index=args.task_id
  )

  if args.job == 'ps':
    # Param server
    server.join()
  elif args.job == 'worker':
    # Start the worker and run the train loop
    run_worker(cluster, server, args)
