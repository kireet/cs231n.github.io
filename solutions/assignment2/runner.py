import numpy as np
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

#number of inputs
#N = 4
N = 4

#number of channels (input depth)
#C = 3
C = 3

#number of filters (output depth)
#F = 2
F = 2

H,W = 5,5
HH,WW = 3,3
x = np.random.randn(N, C, H, W)
w = np.random.randn(F, C, HH, WW)
b = np.random.randn(F,)

pad = 1
stride = 2
conv_param = {'stride': stride, 'pad': pad}

H_out = 1 + (H + 2 * pad - HH) / stride
W_out = 1 + (W + 2 * pad - WW) / stride

print 'output size: (%d,%d)' % (H_out, W_out)
dout = np.random.randn(N, F, H_out, W_out)

dx_num = eval_numerical_gradient_array(lambda x: conv_forward_naive(x, w, b, conv_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_forward_naive(x, w, b, conv_param)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: conv_forward_naive(x, w, b, conv_param)[0], b, dout)

out, cache = conv_forward_naive(x, w, b, conv_param)
dx, dw, db = conv_backward_naive(dout, cache)

# Your errors should be around 1e-9'
print 'Testing conv_backward_naive function'
print 'dx error: ', rel_error(dx, dx_num)
print 'dw error: ', rel_error(dw, dw_num)
print 'db error: ', rel_error(db, db_num)