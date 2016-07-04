from functools import partial

import tensorflow as tf

# Base-class for the Deep Q-Network architecture. Constructs the TensorFlow
# graph with layers, weights, biases, loss-function, optimizer, etc. for
# a network of given type. Currently, a simple network with two hidden layers,
# and a convolutional neural-network are support.
#
# New network architectures can be added by sub-classing Network and
# implmementing the _init_layers() method.
class Network:
  params = []

  x_placeholder = None
  q_placeholder = None
  action_placeholder = None

  q_output = None
  train_op = None
  global_step = None
  summary = None

  def __init__(self, input_shape, num_actions):
    self.input_shape = list(input_shape)
    self.num_actions = num_actions

  @staticmethod
  def create_network(input_shape, num_actions, config):
    Net = {
      'simple': SimpleNetwork,
      'cnn': ConvNetwork,
    }.get(config.network, None)

    if Net is None:
      raise RuntimeError('Unsupported network type {}'.format(config.network))

    net = Net(input_shape, num_actions)
    net._init_network(config)
    return net
  
  def _init_network(self, config):
    # Placeholders
    self.x_placeholder = tf.placeholder(tf.float32, [None] + self.input_shape)
    self.q_placeholder = tf.placeholder(tf.float32, [None])
    self.action_placeholder = tf.placeholder(tf.float32, 
                                             [None, self.num_actions])

    # Inference-loss-training pattern
    self.params, self.q_output, reg_loss = self._init_layers(
      config, 
      inputs=self.x_placeholder, 
      input_shape=self.input_shape, 
      output_size=self.num_actions,
    )
    loss = self._init_loss(
      config, 
      q=self.q_output, 
      expected_q=self.q_placeholder, 
      actions=self.action_placeholder, 
      reg_loss=reg_loss,
    )
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.train_op = self._init_optimizer(
      config,
      params=self.params,
      loss=loss,
      global_step=self.global_step,
    )
    # TODO: Add summary

  @classmethod
  def _init_layers(cls, config, inputs, input_shape, output_size):
    """
    Setup the layers and trainable params of the network. Subclasses should
    implement this to initialize the appropriate network architecture.

    @return: (params, output_layer, regularized_loss)
    """
    raise NotImplementedError
  
  @classmethod
  def _init_loss(cls, config, q, expected_q, actions, reg_loss=None):
    """
    Setup the loss function and apply regularization is provided.

    @return: loss_op
    """
    q_masked = tf.reduce_sum(tf.mul(q, actions), reduction_indices=[1])
    loss = tf.reduce_mean(tf.squared_difference(q_masked, expected_q))
    if reg_loss is not None:
      loss += config.reg_param * reg_loss
    return loss

  @classmethod
  def _init_optimizer(cls, config, params, loss, global_step=None):
    """
    Setup the optimizer for the provided params based on the loss function.
    Relies on config.optimizer to select the type of optimizer.

    @return: train_op
    """

    Optimizer = {
      'adadelta': tf.train.AdadeltaOptimizer,
      'adagrad': tf.train.AdagradOptimizer,
      'adam': tf.train.AdamOptimizer,
      'ftrl': tf.train.FtrlOptimizer,
      'sgd': tf.train.GradientDescentOptimizer,
      'momentum': partial(tf.train.MomentumOptimizer, momentum=config.momentum),
      'rmsprop': partial(tf.train.RMSPropOptimizer, decay=config.rmsprop_decay),
    }.get(config.optimizer, None)

    if Optimizer is None:
      raise RuntimeError('Unsupported optimizer {}'.format(config.optimizer))

    # TODO: Experiment with gating gradients for improved parallelism
    # https://www.tensorflow.org/versions/r0.9/api_docs/python/train.html#gating-gradients
    optimizer = Optimizer(learning_rate=config.lr)

    # Explicitly pass the list of trainable params instead of defaulting to
    # GraphKeys.TRAINABLE_VARIABLES. Otherwise, when this network becomes a
    # subgraph when in-graph replication is configured, TRAINABLE_VARIABLES
    # will contain params from all graph replicas due to global namespacing.
    train_op = optimizer.minimize(
      loss,
      var_list=params,
      global_step=global_step,
    )
    return train_op

# Simple fully connected network with two fully connected layers with
# tanh activations and a final Affine layer.
class SimpleNetwork(Network):
  HIDDEN1_SIZE = 20
  HIDDEN2_SIZE = 20

  @classmethod
  def _init_layers(cls, config, inputs, input_shape, output_size):
    if len(input_shape) != 1:
      raise RuntimeError('%s expects 1-d input' % cls.__class__.__name__)
    input_size = input_shape[0]

    weight_init = tf.truncated_normal_initializer(stddev=0.01)
    bias_init = tf.constant_initializer(value=0.0)

    params = []

    # First hidden layer
    with tf.variable_scope('hidden1'):
      shape = [input_size, cls.HIDDEN1_SIZE]
      w1 = tf.get_variable('w', shape, initializer=weight_init)
      b1 = tf.get_variable('b', cls.HIDDEN1_SIZE, initializer=bias_init)
      a1 = tf.nn.tanh(tf.matmul(inputs, w1) + b1, name='tanh')
      params += [w1, b1]

    # Second hidden layer
    with tf.variable_scope('hidden2'):
      shape = [cls.HIDDEN1_SIZE, cls.HIDDEN2_SIZE]
      w2 = tf.get_variable('w', shape, initializer=weight_init)
      b2 = tf.get_variable('b', cls.HIDDEN2_SIZE, initializer=bias_init)
      a2 = tf.nn.tanh(tf.matmul(a1, w2) + b2, name='tanh')
      params += [w2, b2]
    
    # Output layer
    with tf.variable_scope('output'):
      shape = [cls.HIDDEN2_SIZE, output_size]
      w3 = tf.get_variable('w', shape, initializer=weight_init)
      b3 = tf.get_variable('b', output_size, initializer=bias_init)
      output = tf.add(tf.matmul(a2, w3), b3, name='affine')
      params += [w3, b3]

    # L2 regularization for weights excluding biases
    reg_loss = sum(tf.nn.l2_loss(w) for w in [w1, w2, w3])

    return params, output, reg_loss

# Convolutional network described in
# https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
class ConvNetwork(Network):
  CONV1_FILTERS = 32
  CONV1_SIZE = 8
  CONV1_STRIDE = 4

  CONV2_FILTERS = 64
  CONV2_SIZE = 4
  CONV2_STRIDE = 2

  CONV3_FILTERS = 64
  CONV3_SIZE = 3
  CONV3_STRIDE = 1

  POOL_SIZE = [1, 2, 2, 1]
  POOL_STRIDE = [1, 2, 2, 1]

  FULLY_CONNECTED_SIZE = 256

  @classmethod
  def _init_layers(cls, config, inputs, input_shape, output_size):
    if len(input_shape) != 3:
      raise RuntimeError('%s expects 3-d input' % cls.__class__.__name__)

    weight_init = tf.truncated_normal_initializer(stddev=0.01)
    bias_init = tf.constant_initializer(value=0.0)

    params = []

    # First hidden conv-pool layer
    with tf.variable_scope('conv1'):
      shape = \
        [cls.CONV1_SIZE, cls.CONV1_SIZE, input_shape[2], cls.CONV1_FILTERS]
      w1 = tf.get_variable('w', shape, initializer=weight_init)
      b1 = tf.get_variable('b', cls.CONV1_FILTERS, initializer=bias_init)
      a1 = cls.conv_pool(inputs, w1, b1, cls.CONV1_STRIDE)
      params += [w1, b1]

    # Second hidden conv-pool layer
    with tf.variable_scope('conv2'):
      shape = \
        [cls.CONV2_SIZE, cls.CONV2_SIZE, cls.CONV1_FILTERS, cls.CONV2_FILTERS]
      w2 = tf.get_variable('w', shape, initializer=weight_init)
      b2 = tf.get_variable('b', cls.CONV2_FILTERS, initializer=bias_init)
      a2 = cls.conv_pool(a1, w2, b2, cls.CONV2_STRIDE)
      params += [w2, b2]

    # Third hidden conv-pool layer
    with tf.variable_scope('conv3'):
      shape = \
        [cls.CONV3_SIZE, cls.CONV3_SIZE, cls.CONV2_FILTERS, cls.CONV3_FILTERS]
      w3 = tf.get_variable('w', shape, initializer=weight_init)
      b3 = tf.get_variable('b', cls.CONV3_FILTERS, initializer=bias_init)
      a3 = cls.conv_pool(a2, w3, b3, cls.CONV3_STRIDE)
      params += [w3, b3]

    # Final fully-connected hidden layer
    with tf.variable_scope('fcl'):
      shape = [cls.FULLY_CONNECTED_SIZE, cls.FULLY_CONNECTED_SIZE]
      w4 = tf.get_variable('w', shape, initializer=weight_init)
      b4 = tf.get_variable('b', cls.FULLY_CONNECTED_SIZE, initializer=bias_init)
      a3_flat = tf.reshape(a3, [-1, cls.FULLY_CONNECTED_SIZE])
      a4 = tf.nn.relu(tf.matmul(a3_flat, w4) + b4, name='relu')
      params += [w4, b4]

    with tf.variable_scope('output'):
      shape = [cls.FULLY_CONNECTED_SIZE, output_size]
      w5 = tf.get_variable('w', shape, initializer=weight_init)
      b5 = tf.get_variable('b', output_size, initializer=bias_init)
      output = tf.add(tf.matmul(a4, w5), b5, name='affine')
      params += [w5, b5]

    # L2 regularization for fully-connected weights
    reg_loss = sum(tf.nn.l2_loss(w) for w in [w4, w5])

    return params, output, reg_loss

  @classmethod
  def conv_stride(cls, stride):
    return [1, stride, stride, 1]

  @classmethod
  def conv_pool(cls, inputs, filters, bias, stride):
    conv = tf.nn.conv2d(inputs, filters, strides=cls.conv_stride(stride),
                        padding='SAME', name='conv')
    return cls.max_pool(tf.nn.relu(conv + bias))

  @classmethod
  def max_pool(cls, a):
    return tf.nn.max_pool(a, ksize=cls.POOL_SIZE, strides=cls.POOL_STRIDE,
                          padding='SAME', name='pool')
