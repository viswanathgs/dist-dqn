from collections import deque

import numpy as np

# Stats module for DQNAgent. This is not thread-safe, so one instance
# should be created per agent / train loop.
class Stats:
  def __init__(self):
    self.rewards = deque(maxlen=100)
    self.episodes = 0
    self.total_steps = 0

  def last_100_mean_reward(self):
    return np.mean(self.rewards)

  def log_episode(self, reward, steps):
    self.episodes += 1
    self.rewards.append(reward)
    self.total_steps += steps
