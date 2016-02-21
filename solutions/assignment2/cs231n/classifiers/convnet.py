import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

def describe_solver(solver):
  print '##### optimization options #####'
  print '# epochs: %d' % solver.num_epochs
  print '# batch size: %d' % solver.batch_size
  print '# update rule: %s' % solver.update_rule.__name__
  print '# lr_decay: %f' % solver.lr_decay
  for k,v in sorted(solver.optim_config.iteritems()):
    print '# cfg.%s: %s' % (k, str(v))
  print '################################'

class ConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=True, gradcheck=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################

    #conv input dimensions
    h1 = input_dim[1]
    w1 = input_dim[2]
    padding = (filter_size - 1) / 2
    stride = 1

    #conv output dimensions - relu/pool input dimensions
    h2 = (h1 - filter_size + 2 * padding)/stride + 1
    w2 = (w1 - filter_size + 2 * padding)/stride + 1

    #relu/pool output dimensions (pool has stride of 2)
    h3 = h2/2
    w3 = w2/2

    self.params['W1'] = np.random.normal(scale=weight_scale, size=(num_filters, input_dim[0], filter_size, filter_size))
    self.params['b1'] = np.zeros(num_filters)

    if use_batchnorm:
      self.params['gamma1'] = np.ones(num_filters)
      self.params['beta1'] = np.zeros(num_filters)

    self.params['W2'] = np.random.normal(scale=weight_scale, size=(h3*w3*num_filters, hidden_dim))

    if gradcheck: #avoid relu non-linearity
      self.params['b2'] = np.ones(hidden_dim)
    else:
      self.params['b2'] = np.zeros(hidden_dim)

    if use_batchnorm:
      self.params['gamma2'] = np.ones(hidden_dim)
      self.params['beta2'] = np.zeros(hidden_dim)

    self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)

    if use_batchnorm:
      self.bn_params = [{'mode': 'train'}, {'mode': 'train'}]

    self.use_batchnorm = use_batchnorm
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    self.print_model_desc(weight_scale, gradcheck)


  def print_model_desc(self, weight_scale, gradcheck):
    print '***** MODEL CONFIGURATION *****'
    if gradcheck:
        print '* WARNING: in gradient check mode'

    print '* initialization weight scale: ' + str(weight_scale)
    if self.use_batchnorm:
        print '* batch normalization enabled'
    else:
        print '* batch normalization DISABLED'

    if self.reg == 0:
      print '* L2 regularization DISABLED'
    else:
      print '* L2 regularization ' + str(self.reg)

    print '*\n* layers:'
    i = 1
    while True:
      weights = self.params.get('W' + str(i))
      last = self.params.get('W' + str(i+1)) is None
      if weights is None:
        break

      if len(weights.shape) == 4:
        F, _, H, W = weights.shape
        print '*\tCONV layer: %dx%d filter, depth %d' % (H,W,F)
      elif len(weights.shape) == 2:
        D = weights.shape[1]
        layer_type = 'FINAL AFFINE/SOFTMAX' if last else 'AFFINE_RELU'
        units_type = 'classes' if last else 'hidden relu units'
        print '*\t%s layer: %d %s' % (layer_type, D, units_type)
      else:
        raise ValueError('unrecognized layer type')

      i += 1

    print '*\n* weight dimensions:'
    for k, v in sorted(self.params.iteritems()):
      print '*\t%s: %s' % (k, v.shape)
      self.params[k] = v.astype(self.dtype)

    print '*******************************'


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """

    if self.use_batchnorm:
      mode = 'test' if y is None else 'train'
      for bn_param in self.bn_params:
        bn_param['mode'] = mode

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out, conv_cache = conv_relu_pool_forward(X, self.params['W1'], self.params['b1'], conv_param, pool_param)
    if self.use_batchnorm:
      out, bn1_cache = spatial_batchnorm_forward(out, self.params['gamma1'], self.params['beta1'], self.bn_params[0])

    out, hidden_cache = affine_relu_forward(out, self.params['W2'], self.params['b2'])

    if self.use_batchnorm:
      out, bn2_cache = batchnorm_forward(out, self.params['gamma2'], self.params['beta2'], self.bn_params[1])


    scores, output_cache = affine_forward(out, self.params['W3'], self.params['b3'])

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores

    loss, grads = 0, {}

    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores, y)
    grads = {}
    reg = self.reg

    loss += .5 * reg * np.sum(self.params['W3']**2)
    loss += .5 * reg * np.sum(self.params['W2']**2)
    loss += .5 * reg * np.sum(self.params['W1']**2)

    dout, grads['W3'], grads['b3'] = affine_backward(dout, output_cache)
    grads['W3'] += reg * self.params['W3']

    if self.use_batchnorm:
      dout, grads['gamma2'], grads['beta2'] = batchnorm_backward(dout, bn2_cache)

    dout, grads['W2'], grads['b2'] = affine_relu_backward(dout, hidden_cache)
    grads['W2'] += reg * self.params['W2']

    if self.use_batchnorm:
      dout, grads['gamma1'], grads['beta1'] = spatial_batchnorm_backward(dout, bn1_cache)

    dout, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout, conv_cache)
    grads['W1'] += reg * self.params['W1']


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
