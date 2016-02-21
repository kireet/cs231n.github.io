import numpy as np
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver
from cs231n.classifiers.convnet import *

# def rel_error(x, y):
#   """ returns relative error """
#   return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
#
# np.random.seed(0)
#
# N = 50
# X = np.random.randn(N, 3, 8, 8)
# y = np.random.randint(10, size=N)
#
# input_dim = (X.shape[1], X.shape[2], X.shape[3])
#
# model = ConvNet(num_filters=2, input_dim=input_dim, filter_size=5, hidden_dim=10, use_batchnorm=True, gradcheck=True)
#
# #model = ThreeLayerConvNet(num_filters=2, input_dim=input_dim, filter_size=5, hidden_dim=10)
#
# loss, grads = model.loss(X, y)
# print 'Initial loss (no regularization): ', loss
#
# for name in sorted(grads):
#   f = lambda _: model.loss(X, y)[0]
#   grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
#   print '%s relative error: %.2e' % (name, rel_error(grad_num, grads[name]))

data = get_CIFAR10_data()
#for k, v in data.iteritems():
#  print '%s: ' % k, v.shape

model = ConvNet(weight_scale=0.001, hidden_dim=500, reg=0)

print

solver = Solver(model, data,
                num_epochs=10, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 5e-3,
                },
                verbose=True, print_every=100)

describe_solver(solver)

print

solver.train()