from collections import deque
from six.moves import map

import random
import utils

# Replay memory to store state observations from the environment.
# Implmemented as a deque with a fixed amount of maximum capacity.
# Provides the ability to randomly sample a minibatch from the memory.
#
# Thread-safety: collections.deque is thread-safe for append() and pop(),
# so adding new entries into the replay memory from multiple threads is safe.
# But get_minibatch() is currently not thread-safe.

# TODO: Add thread-safety for get_minibatch() via a reader-writer lock.
# This would allow replay memory to be shared across multiple worker threads
# controlling different devices in the in-graph data-parallelism model.
class ReplayMemory:
  def __init__(self, capacity): 
    self.memory = deque(maxlen=capacity)

  def add(self, state, action, reward, observation, terminal):
    self.memory.append((state, action, reward, observation, terminal))

  def size(self):
    return len(self.memory)

  def capacity(self):
    return self.memory.maxlen

  def get_minibatch(self, minibatch_size):
    """
    Samples a minibatch of configured size from the memory, splits the
    minibatch into two partitions based on the observation being in terminal
    state or not. 

    @return: (non_terminal_minibatch, terminal_minibatch)

    If memory size is smaller than the configure minibatch size, 
    returns (None, None).
    """
    if self.size() < minibatch_size:
      return None, None
    minibatch = random.sample(self.memory, minibatch_size)
    return utils.partition(lambda x: x[4], minibatch)

  @staticmethod
  def get_states(iterable):
    return map(lambda m: m[0], iterable)

  @staticmethod
  def get_next_states(iterable):
    return map(lambda m: m[3], iterable)
