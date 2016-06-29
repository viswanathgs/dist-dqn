from replay_memory import ReplayMemory
from six.moves import range, zip, zip_longest
from stats import Stats

import itertools
import logging
import random
import tensorflow as tf
import utils

class DQNAgent:
  FAIURE_REWARD = {
    'CartPole-v0': -100,
  }

  def __init__(self, env, network, session, replay_memory, config):
    self.env = env
    self.network = network
    self.session = session
    self.replay_memory = replay_memory
    self.config = config

    self.random_action_prob = config.init_random_action_prob
    self.total_steps = 0
    self.stats = Stats()

    # Initialize target network
    self._update_target_network()

  def train(self, num_episodes, max_steps_per_episode):
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
        'Episode = %d, steps = %d, total steps = %d, reward = %d, '
        'last-100 mean reward = %.2f' %
        (episode, steps, self.total_steps, reward, mean_reward))

      # Update target network if needed
      if self.total_steps % self.config.target_update_freq == 0:
        self._update_target_network()

  def train_episode(self, max_steps):
    """
    Run one episode of the gym environment, add transitions to replay memory,
    and train minibatches from replay memory against the target network.
    """
    state = self.env.reset()
    total_reward = 0

    # TODO: Add support for multiple steps to capture time/motion for Pong
    done = False
    steps = 0
    while not done and (steps < max_steps):
      # Pick the next action and execute it
      action = self._pick_action(state)
      observation, reward, done, _ = self.env.step(action)
      total_reward += reward

      # Punish hard on failure
      if done:
        reward = self.FAIURE_REWARD.get(self.env.spec.id, reward)
      # TODO: Implement reward clipping

      # Add the transition to replay memory and train a sampled minibatch
      self.replay_memory.insert(state, action, reward, observation, done)
      self._train_minibatch(self.config.minibatch_size)

      state = observation
      steps += 1
      self.total_steps += 1
    return total_reward, steps

  def _train_minibatch(self, minibatch_size):
    if self.replay_memory.size() < minibatch_size:
      return

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
    # TODO: Add TensorFlow summary
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
    self.random_action_prob = utils.decay(
      val=self.random_action_prob,
      min_val=self.config.min_random_action_prob, 
      decay_rate=self.config.random_action_prob_decay,
    )

  def _predict_q_values(self, states, use_target_network=False):
    """
    Run forward-prop through the network and fetch the q-values.
    If use_target_network is True, then target_params will be used for
    forward-prop.

    @return: Numpy array of q-values for each state
    """
    feed_dict = {
      self.network.x_placeholder: states,
    }
    if use_target_network:
      feed_dict.update(zip(self.network.params, self.target_params))
    return self.session.run(self.network.q_output, feed_dict=feed_dict)

  def _update_target_network(self):
    """
    Update the target network by capturing the current state of the
    network params.
    """
    # TODO: This is slow as the params need to be copied from the device
    # to the client / host memory. The target network could be made part
    # of the graph itself using TensorFlow's control flow operators. That
    # would make sure the target network always resides within the device
    # and would avoid having to feed the weights to the device along 
    # with minibatch for each gradient update step.
    self.target_params = self.session.run(self.network.params)

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
