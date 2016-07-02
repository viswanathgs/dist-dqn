import cv2
import itertools
import numpy as np

def partition(pred, iterable):
  """
  Partition the iterable into two disjoint entries based
  on the predicate.

  @return: Tuple (iterable1, iterable2)
  """
  iter1, iter2 = itertools.tee(iterable)
  return itertools.filterfalse(pred, iter1), filter(pred, iter2)

def decay(val, min_val, decay_rate):
  return max(val * decay_rate, min_val)

def one_hot(i, n):
  """
  One-hot encoder. Returns a numpy array of length n with i-th entry
  set to 1, and all others set to 0."

  @return: numpy.array
  """
  assert i < n, "Invalid args to one_hot"
  enc = np.zeros(n)
  enc[i] = 1
  return enc

def resize_image(image, width, height):
  """
  Resize the image screen to the configured width and height and
  convert it to grayscale.
  """
  grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  return cv2.resize(grayscale, (width, height))
