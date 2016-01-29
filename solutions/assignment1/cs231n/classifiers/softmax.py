import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  loss = 0
  for i in xrange(num_train):
    f = X[i].dot(W)
    f -= np.max(f)

    exp_f = np.exp(f)
    sum_exp_f = np.sum(exp_f)

    pred = exp_f / sum_exp_f
    loss -= np.log(pred[y[i]])

    for j in xrange(num_classes):
      mult = pred[j]
      if j == y[i]:
        mult -= 1
      dW[:,j] += X[i] * mult

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W

  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  m = X.shape[0]

  F = X.dot(W) # NxC
  F = F - np.reshape(np.max(F, axis=1), (m,1))
  P = np.exp(F) # NxC
  P = P / np.sum(P, axis=1).reshape((m,1))
  loss = np.sum( -np.log(P[xrange(m),y]) )

  P[xrange(m),y] -= 1
  dW = X.T.dot(P)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  loss /= m
  loss += 0.5 * reg * np.sum(W * W)
  dW /= m
  dW += reg * W

  return loss, dW

