import os

import numpy as np

import representation.mnist as mnist
import representation.cifar as cifar
import representation.cifar100 as cifar100


def estimates_mnist():
  output_dir = 'outputs/one_point_estimates/'
  (x_train, y_train), (x_test, y_test) = mnist.input_data()
  model = mnist.build_model(10)
  model_path = 'saved_models/mnist.h5'
  model.load_weights(model_path)
  # Predict
  proba = model.predict(x_test)
  # Saving the probabilities
  np.save(os.path.join(output_dir, 'p_mnist.npy'), proba)
  
  
def estimates_cifar10():
  output_dir = 'outputs/one_point_estimates/'
  (x_train, y_train), (x_test, y_test) = cifar.input_cifar10()
  model = cifar.build_model(x_train, 10)
  model_path = 'saved_models/keras_cifar10_trained_model.h5'
  model.load_weights(model_path)
  # Predict
  proba = model.predict(x_test)
  # Saving the probabilities
  np.save(os.path.join(output_dir, 'p_cifar10.npy'), proba)
  
  
def estimates_cifar100():
  output_dir = 'outputs/one_point_estimates/'
  (x_train, y_train), (x_test, y_test) = cifar.input_cifar100()
  model = cifar100.build_model(x_train, 100)
  model_path = 'saved_models/andrewkruger_cifar100.h5'
  model.load_weights(model_path)
  # Predict
  proba = model.predict(x_test)
  # Saving the probabilities
  np.save(os.path.join(output_dir, 'p_cifar100.npy'), proba)
  
estimates_mnist()
estimates_cifar10()
estimates_cifar100()