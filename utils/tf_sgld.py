# -*- coding: utf-8 -*-
""" Implementation of SGLD in term of tf.Keras optimizer."""

#%% Packages

import tensorflow.keras as keras
K = keras.backend
    
#%% class SGLD

class SGLD(keras.optimizers.Optimizer):
  """Stochastic gradient descent optimizer.
  Includes support for learning rate decay.
  # Arguments
      N: int. Number of training data items.
      lr: float >= 0. Learning rate.
      decay: float >= 0. Learning rate decay over each update.
  """

  def __init__(self, N, lr=0.01, decay=0., **kwargs):
      super(SGLD, self).__init__(**kwargs)
      with K.name_scope(self.__class__.__name__):
          self.iterations = K.variable(0, dtype='int64', name='iterations')
          self.lr = K.variable(lr, name='lr')
          self.decay = K.variable(decay, name='decay')
      self.initial_decay = decay
      self.N = N

  def get_updates(self, loss, params):
      grads = self.get_gradients(loss, params)
      self.updates = [K.update_add(self.iterations, 1)]

      lr = self.lr
      if self.initial_decay > 0:
          lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                    K.dtype(self.decay))))
      shapes = [K.int_shape(p) for p in params]
      gaussian = [K.random_normal(shape) for shape in shapes]
#      self.weights = [self.iterations] + moments
      for p, g, gaus in zip(params, grads, gaussian):
          v = - lr * g  # velocity
          n = K.sqrt(2.0 * lr / self.N) * gaus # gaussian noise
          new_p = p + v + n

          # Apply constraints.
          if getattr(p, 'constraint', None) is not None:
              new_p = p.constraint(new_p)

          self.updates.append(K.update(p, new_p))
      return self.updates

  def get_config(self):
      config = {'lr': float(K.get_value(self.lr)),
                'decay': float(K.get_value(self.decay)),
                'N': self.N
               }
      base_config = super(SGLD, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))