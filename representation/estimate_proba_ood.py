import os

import numpy as np

import representation.mnist as mnist
import representation.cifar as cifar
import representation.cifar100 as cifar100


def estimates_mnist():
  output_dir = 'outputs_ood/one_point_estimates/mnist_sec_5'
  (x_train, y_train), (x_test, y_test) = mnist.input_data()

  index_path = 'saved_models_ood/mnist_sec_5/index.npy'
  index = np.load(index_path)
  n_class = np.sum(index)
  indices_test_in = np.dot(y_test, index).astype(bool)

  model = mnist.build_model(n_class)
  model_path = 'saved_models_ood/mnist_sec_5/mnist.h5'
  model.load_weights(model_path)

  # Predict
  proba_in = model.predict(x_test[indices_test_in, :])
  proba_out = model.predict(x_test[~indices_test_in, :])

  # Saving the probabilities
  np.save(os.path.join(output_dir, 'p_mnist_in.npy'), proba_in)
  np.save(os.path.join(output_dir, 'p_mnist_out.npy'), proba_out)
  
  
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
#estimates_cifar10()
#estimates_cifar100()