from collections import deque

import numpy as np

# A circular buffer implemented as a deque to keep track of the last few
# frames in the environment that together form a state capturing temporal
# and directional information. Provides an accessor to get the current
# state at any given time, which is represented as a list of consecutive
# frames.
#
# Also takes in a pre-processor to potentially resize or modify the frames
# before inserting them into the buffer.
class FrameBuffer:
  def __init__(self, frames_per_state, preprocessor=lambda x: x):
    """
    @param frames_per_state:  Number of consecutive frames that form a state.
    @param reprocessor:       Lambda that takes a frame and returns a
                              preprocessed frame.
    """
    if frames_per_state <= 0:
      raise RuntimeError('Frames per state should be greater than 0')

    self.frames_per_state = frames_per_state
    self.frames = deque(maxlen=frames_per_state)
    self.preprocessor = preprocessor

  def append(self, frame):
    """
    Takes a frame, applies preprocessing, and appends it to the deque.

    The first frame added to the buffer is duplicated `frames_per_state` times
    to completely fill the buffer.
    """
    frame = self.preprocessor(frame)
    if len(self.frames) == 0:
      self.frames.extend(self.frames_per_state * [frame])
    else:
      self.frames.append(frame)

  def get_state(self):
    """
    Fetch the current state consisting of `frames_per_state` consecutive frames.

    If `frames_per_state` is 1, returns the frame instead of an array of
    length 1. Otherwise, returns a Numpy array of `frames_per_state` frames.
    """
    if len(self.frames) == 0:
      return None
    if self.frames_per_state == 1:
      return self.frames[0]
    return np.stack(self.frames, axis=-1)

  def clear(self):
    """
    Clear the frames in the buffer.
    """
    self.frames.clear()
