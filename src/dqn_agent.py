from frame_buffer import FrameBuffer
from functools import partial
from replay_memory import ReplayMemory
from six.moves import range, zip, zip_longest
from stats import Stats

import itertools
import logging
import random
import tensorflow as tf
import utils

class DQNAgent:
  # Environments for which the state frames need to be resized
  RESIZE_STATE_FRAMES = [
    'Pong-v0',
  ]

  # Reward penalty on failure for each environment
  FAILURE_PENALTY = {
    'CartPole-v0': -100,
  }

  def __init__(self, env, network, session, replay_memory, config,
               enable_summary=True):
    self.env = env
    self.network = network
    self.session = session
    self.replay_memory = replay_memory
    self.config = config

    self.training_steps = 0 # Keeps count of learning updates
    self.stats = Stats()

    self.random_action_prob = config.init_random_action_prob
    self.random_action_prob_decay = utils.decay_per_step(
      init_val=self.config.init_random_action_prob,
      min_val=self.config.min_random_action_prob,
      steps=self.config.random_action_explore_steps,
    )

    self.summary_writer = None
    if enable_summary:
      self.summary_writer = tf.train.SummaryWriter(config.logdir, session.graph)

    self.frame_buffer = FrameBuffer(
      frames_per_state=config.frames_per_state,
      preprocessor=self._get_frame_resizer(env, config),
    )

    # Prefill the replay memory with experiences based on random actions
    self._prefill_replay_memory(self.config.replay_start_size)

    # Initialize the target network
    self._update_target_network()

  def train(self, num_episodes, max_steps_per_episode, supervisor=None):
    """
    Train the DQN for the configured number of episodes.
    """
    for episode in range(num_episodes):
      # Train an episode
      reward, steps = self.train_episode(max_steps_per_episode)

      # Update stats
      self.stats.log_episode(reward, steps)
      mean_reward = self.stats.last_100_mean_reward()
      logging.info(
        'Episode = %d, steps = %d, reward = %d, training steps = %d, '
        'last-100 mean reward = %.2f' %
        (episode, steps, reward, self.training_steps, mean_reward))

      if supervisor and supervisor.should_stop():
        logging.warning('Received signal to stop. Exiting train loop.')
        break

  def train_episode(self, max_steps):
    """
    Run one episode of the gym environment, add transitions to replay memory,
    and train minibatches from replay memory against the target network.
    """
    self.frame_buffer.clear()
    observation = self.env.reset()
    self.frame_buffer.append(observation)
    state = self.frame_buffer.get_state()

    total_reward = steps = 0
    done = False
    while not done and (steps < max_steps):
      # Pick the next action and execute it
      action = self._pick_action(state)
      observation, reward, done, _ = self.env.step(action)
      total_reward += reward
      steps += 1

      # Punish hard on failure
      if done:
        reward = self.FAILURE_PENALTY.get(self.env.spec.id, reward)
      # TODO: Implement reward clipping

      # Add the transition to replay memory and update the current state
      self.frame_buffer.append(observation)
      next_state = self.frame_buffer.get_state()
      self.replay_memory.add(state, action, reward, next_state, done)
      state = next_state

      # Train a minibatch
      if steps % self.config.update_freq == 0:
        self._train_minibatch(self.config.minibatch_size)

      # Update the target network if needed
      self._update_target_network()

    return total_reward, steps

  def _train_minibatch(self, minibatch_size):
    if self.replay_memory.size() < minibatch_size:
      return

    self.training_steps += 1

    # Sample a minibatch from replay memory
    non_terminal_minibatch, terminal_minibatch = \
                    self.replay_memory.get_minibatch(minibatch_size)
    non_terminal_minibatch, terminal_minibatch = \
                    list(non_terminal_minibatch), list(terminal_minibatch)

    # Compute max q-values for the non-terminal next states based
    # on the target network
    next_states = list(ReplayMemory.get_next_states(non_terminal_minibatch))
    q_values = self._predict_q_values(next_states, use_target_network=True)
    max_q_values = q_values.max(axis=1)

    # Gradient descent
    feed_dict = self._get_minibatch_feed_dict(
      max_q_values,
      non_terminal_minibatch,
      terminal_minibatch,
    )

    if self._should_log_summary():
      _, summary = self.session.run(
        [self.network.train_op, self.network.summary_op],
        feed_dict=feed_dict,
      )
      self.summary_writer.add_summary(summary, self.training_steps)
    else:
      self.session.run(self.network.train_op, feed_dict=feed_dict)

  def _pick_action(self, state):
    """
    Pick the next action given the current state.

    Based on a biased dice roll, either a random action, or the
    action corresponding to the max q-value obtained by executing 
    forward-prop is chosen.

    @return: action
    """
    if self._roll_random_action_dice():
      return self.env.action_space.sample()

    # Run forward prop and return the action with max q-value
    q_values = self._predict_q_values([state])
    return q_values.argmax()
  
  def _roll_random_action_dice(self):
    """
    Roll the dice based on the configured probability, as well as decay
    the probability.

    @return: True if random action should be chosen, False otherwise.
    """
    self._decay_random_action_prob()
    return random.random() < self.random_action_prob

  def _decay_random_action_prob(self):
    if self.random_action_prob > self.config.min_random_action_prob:
      self.random_action_prob -= self.random_action_prob_decay

  def _predict_q_values(self, states, use_target_network=False):
    """
    Run forward-prop through the network and fetch the q-values.
    If use_target_network is True, then the target network's params
    will be used for forward-prop.

    @return: Numpy array of q-values for each state
    """
    q_output = self.network.target_q_output if use_target_network \
                                            else self.network.q_output
    feed_dict = {
      self.network.x_placeholder: states,
    }
    return self.session.run(q_output, feed_dict=feed_dict)

  def _prefill_replay_memory(self, prefill_size):
    """
    Prefill the replay memory by picking actions via uniform random policy,
    executing them, and adding the experiences to the memory.
    """
    terminal = True
    while self.replay_memory.size() < prefill_size:
      # Reset the environment and the frame buffer between gameplays
      if terminal:
        self.frame_buffer.clear()
        observation = self.env.reset()
        self.frame_buffer.append(observation)
        state = self.frame_buffer.get_state()

      # Sample a random action and execute it
      action = self.env.action_space.sample()
      observation, reward, terminal, _ = self.env.step(action)

      # Pre-populate replay memory with the experience
      self.frame_buffer.append(observation)
      next_state = self.frame_buffer.get_state()
      self.replay_memory.add(state, action, reward, next_state, terminal)
      state = next_state

  def _update_target_network(self):
    """
    Update the target network by capturing the current state of the
    network params.
    """
    if self.training_steps % self.config.target_update_freq == 0:
      logging.info('Updating target network')
      self.session.run(self.network.target_update_ops)

  def _get_minibatch_feed_dict(self, target_q_values, 
                               non_terminal_minibatch, terminal_minibatch):
    """
    Helper to construct the feed_dict for train_op. Takes the non-terminal and 
    terminal minibatches as well as the max q-values computed from the target
    network for non-terminal states. Computes the expected q-values based on
    discounted future reward.

    @return: feed_dict to be used for train_op
    """
    assert len(target_q_values) == len(non_terminal_minibatch)

    states = []
    expected_q = []
    actions = []

    # Compute expected q-values to plug into the loss function
    minibatch = itertools.chain(non_terminal_minibatch, terminal_minibatch)
    for item, target_q in zip_longest(minibatch, target_q_values, fillvalue=0):
      state, action, reward, _, _ = item
      states.append(state)
      # target_q will be 0 for terminal states due to fillvalue in zip_longest
      expected_q.append(reward + self.config.reward_discount * target_q)
      actions.append(utils.one_hot(action, self.env.action_space.n))

    return {
      self.network.x_placeholder: states, 
      self.network.q_placeholder: expected_q,
      self.network.action_placeholder: actions,
    }

  def _should_log_summary(self):
    if self.summary_writer is None:
      return False

    summary_freq = self.config.summary_freq
    return summary_freq > 0 and (self.training_steps % summary_freq == 0)

  @classmethod
  def _get_frame_resizer(cls, env, config):
    """
    Returns a lambda that takes a screen frame and resizes it to the
    configured width and height. If the state doesn't need to be resized
    for the environment, returns an identity function.

    @return: lambda (frame -> resized_frame)
    """
    if env.spec.id not in cls.RESIZE_STATE_FRAMES:
      return lambda x: x
    return partial(
      utils.resize_image,
      width=config.resize_width,
      height=config.resize_height,
    )

  @classmethod
  def get_input_shape(cls, env, config):
    """
    Return the shape of the input to the network based on the environment,
    config, and whether screen frames need to be resized or not.
    """
    if env.spec.id not in cls.RESIZE_STATE_FRAMES:
      return env.observation_space.shape
    return (config.resize_width, config.resize_height, config.frames_per_state)
