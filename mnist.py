# -*- coding: utf-8 -*-

#%% Packages

# TensorFlow and tf.keras
import tensorflow as tf
K = tf.keras.backend

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
    
#%% class SGLD

class SGLD(tf.keras.optimizers.Optimizer):
  """Stochastic gradient descent optimizer.
  Includes support for momentum,
  learning rate decay.
  # Arguments
      lr: float >= 0. Learning rate.
      decay: float >= 0. Learning rate decay over each update.
  """

  def __init__(self, lr=0.01, decay=0., **kwargs):
      super(SGLD, self).__init__(**kwargs)
      with K.name_scope(self.__class__.__name__):
          self.iterations = K.variable(0, dtype='int64', name='iterations')
          self.lr = K.variable(lr, name='lr')
          self.decay = K.variable(decay, name='decay')
      self.initial_decay = decay

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
          noise = K.sqrt(2 * lr) * gaus # gaussian noise
          # Error: divide the step size by 1/N for the Gaussian noise.
          new_p = p + v # + noise

          # Apply constraints.
          if getattr(p, 'constraint', None) is not None:
              new_p = p.constraint(new_p)

          self.updates.append(K.update(p, new_p))
      return self.updates

  def get_config(self):
      config = {'lr': float(K.get_value(self.lr)),
                'decay': float(K.get_value(self.decay))
               }
      base_config = super(SGLD, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))

sgld = SGLD()

#%% Mnist

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(20, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=sgld,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20)
losses = model.evaluate(x_test, y_test)
