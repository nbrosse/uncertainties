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
  output_dir = 'outputs_ood/one_point_estimates/cifar10_sec_5'
  (x_train, y_train), (x_test, y_test) = cifar.input_cifar10()
  
  index_path = 'saved_models_ood/cifar10_sec_5/index.npy'
  index = np.load(index_path)
  n_class = np.sum(index)
  indices_test_in = np.dot(y_test, index).astype(bool)
  
  model = cifar.build_model(x_train, n_class)
  model_path = 'saved_models_ood/cifar10_sec_5/keras_cifar10_trained_model.h5'
  model.load_weights(model_path)
  # Predict
  proba_in = model.predict(x_test[indices_test_in, :])
  proba_out = model.predict(x_test[~indices_test_in, :])
  # Saving the probabilities
  np.save(os.path.join(output_dir, 'p_cifar10_in.npy'), proba_in)
  np.save(os.path.join(output_dir, 'p_cifar10_out.npy'), proba_out)
  
  
def estimates_cifar100():
  output_dir = 'outputs_ood/one_point_estimates/cifar100_sec_50'
  (x_train, y_train), (x_test, y_test) = cifar.input_cifar100()
  
  index_path = 'saved_models_ood/cifar100_sec_50/index.npy'
  index = np.load(index_path)
  n_class = np.sum(index)
  indices_test_in = np.dot(y_test, index).astype(bool)
  
  model = cifar100.build_model(x_train, n_class)
  model_path = 'saved_models_ood/cifar100_sec_50/andrewkruger_cifar100.h5'
  model.load_weights(model_path)
  # Predict
  proba_in = model.predict(x_test[indices_test_in, :])
  proba_out = model.predict(x_test[~indices_test_in, :])
  # Saving the probabilities
  np.save(os.path.join(output_dir, 'p_cifar100_in.npy'), proba_in)
  np.save(os.path.join(output_dir, 'p_cifar100_out.npy'), proba_out)
  
#estimates_mnist()
#estimates_cifar10()
estimates_cifar100()