#%% Imports

import os
from absl import app, flags
import pickle

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', None, 'mnist, cifar10 or cifar100')

import numpy as np
import keras

import full_network_algos.full_network as full_network
import utils.sgld as sgld
import representation.mnist as mnist
import representation.cifar as cifar
import representation.cifar100 as cifar100
import utils.util as util

#%% Launch functions

def launch_mnist_sgd_sgld(epochs, batch_size, thinning_interval, lr):
  output_dir = util.create_run_dir('outputs/full_network/sgd_sgld/mnist')
  (x_train, y_train), (x_test, y_test) = mnist.input_data()
  model = mnist.build_model()
  model_path = 'saved_models/mnist.h5'
  model.load_weights(model_path, by_name=True)
  path_dic = {}
  # Create metrics dir
  path_metrics = os.path.join(output_dir, 'metrics')
  os.makedirs(path_metrics)
  # Sampling
  for opt, optimizer in zip(['sgd', 'sgld'], 
                            [keras.optimizers.SGD(lr=lr), 
                             sgld.SGLD(x_train.shape[0], lr=lr)]):
    path_weights = os.path.join(output_dir, opt)
    os.makedirs(path_weights)
    hist = full_network.sgd_sgld(model, optimizer, epochs, 
                                 batch_size, x_train, 
                                 y_train, x_test, 
                                 y_test, thinning_interval, 
                                 path_weights)
    # Save history of the training
    with open(os.path.join(path_metrics, 
                           'hist_{}.pkl'.format(opt)), 'wb') as handle:
      pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    path_dic[opt] = path_weights
    print('End of sampling for %s' % opt)
  print('End of sampling.')
  # Compute the probabilities
  for opt in ['sgd', 'sgld']:
    proba_tab = full_network.predict_sgd_sgld(model, x_test,
                                              10, path_dic[opt])
    np.save(os.path.join(path_metrics, 'p_{}.npy'.format(opt)), proba_tab)
  print('End of computing probabilities')


def launch_mnist_dropout(epochs, batch_size, p_dropout, num_samples):
  output_dir = util.create_run_dir('outputs/full_network/dropout/mnist')
  (x_train, y_train), (x_test, y_test) = mnist.input_data()
  model = full_network.build_model_mnist(p_dropout)
  model_path = 'saved_models/mnist.h5'
  model.load_weights(model_path, by_name=True)
  # Create metrics dir
  path_metrics = os.path.join(output_dir, 'metrics')
  os.makedirs(path_metrics)
  # Training
  model.compile(optimizer='sgd', 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
  hist = model.fit(x_train, y_train,
                   batch_size=batch_size,
                   epochs=epochs,
                   verbose=1,
                   validation_data=(x_test, y_test))
  # Saving the model
  model.save(os.path.join(output_dir, 'weights.h5'))
  # Save history of the training
  with open(os.path.join(path_metrics, 'hist_dropout.pkl'), 'wb') as handle:
      pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
  # Sampling
  proba_tab = np.zeros(shape=(x_test.shape[0], 10, num_samples))
  for index in np.arange(num_samples):
    proba = model.predict(x_test)
    proba_tab[:, :, index] = proba
  # Saving the probabilities
  np.save(os.path.join(path_metrics, 'p_dropout.npy'), proba_tab)

#%% Cifar
  
def launch_cifar_sgd_sgld(epochs, batch_size, thinning_interval, lr):
  output_dir = util.create_run_dir('outputs/full_network/'
                                   'sgd_sgld/{}'.format(FLAGS.dataset))
  
  if FLAGS.dataset == 'cifar10':
    (x_train, y_train), (x_test, y_test) = cifar.input_cifar10()
    model = cifar.build_model(x_train, 10)
    model_path = 'saved_models/keras_cifar10_trained_model.h5'
    num_classes = 10
  else:
    (x_train, y_train), (x_test, y_test) = cifar.input_cifar100()
    model = cifar100.build_model(x_train, 100)
    model_path = 'saved_models/andrewkruger_cifar100.h5'
    num_classes = 100
  
  model.load_weights(model_path, by_name=True)
  path_dic = {}
  # Create metrics dir
  path_metrics = os.path.join(output_dir, 'metrics')
  os.makedirs(path_metrics)
  # Sampling
  for opt, optimizer in zip(['sgd', 'sgld'], 
                            [keras.optimizers.SGD(lr=lr), 
                             sgld.SGLD(x_train.shape[0], lr=lr)]):
    path_weights = os.path.join(output_dir, opt)
    os.makedirs(path_weights)
    hist = full_network.sgd_sgld(model, optimizer, epochs, 
                                 batch_size, x_train, 
                                 y_train, x_test, 
                                 y_test, thinning_interval, 
                                 path_weights)
    # Save history of the training
    with open(os.path.join(path_metrics, 
                           'hist_{}.pkl'.format(opt)), 'wb') as handle:
      pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    path_dic[opt] = path_weights    
    print('End of sampling for %s' % opt)
  print('End of sampling.')
  # Compute the probabilities
  for opt in ['sgd', 'sgld']:
    proba_tab = full_network.predict_sgd_sgld(model, x_test,
                                              num_classes, 
                                              path_dic[opt])
    np.save(os.path.join(path_metrics, 'p_{}.npy'.format(opt)), proba_tab)
  print('End of computing probabilities')


def launch_cifar_dropout(epochs, batch_size, p_dropout, num_samples):
  output_dir = util.create_run_dir('outputs/full_network/'
                                   'dropout/{}'.format(FLAGS.dataset))
  
  if FLAGS.dataset == 'cifar10':
    (x_train, y_train), (x_test, y_test) = cifar.input_cifar10()
    model = full_network.build_model_cifar10(x_train, 10, p_dropout)
    model_path = 'saved_models/keras_cifar10_trained_model.h5'
    num_classes = 10
  else:
    (x_train, y_train), (x_test, y_test) = cifar.input_cifar100()
    model = full_network.build_model_cifar100(x_train, 100, p_dropout)
    model_path = 'saved_models/andrewkruger_cifar100.h5'
    num_classes = 100
  
  model.load_weights(model_path, by_name=True)
  
  # Create metrics dir
  path_metrics = os.path.join(output_dir, 'metrics')
  os.makedirs(path_metrics)
  # Training
  model.compile(optimizer='sgd', 
                   loss='categorical_crossentropy', 
                   metrics=['accuracy'])
  hist = model.fit(x_train, y_train,
                   batch_size=batch_size,
                   epochs=epochs,
                   verbose=1,
                   validation_data=(x_test, y_test))
  # Saving the model
  model.save(os.path.join(output_dir, 'weights.h5'))
  # Save history of the training
  with open(os.path.join(path_metrics, 'hist_dropout.pkl'), 'wb') as handle:
      pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)  
  # Sampling
  proba_tab = np.zeros(shape=(x_test.shape[0], num_classes, 
                              num_samples))
  for index in np.arange(num_samples):
    proba = model.predict(x_test)
    proba_tab[:, :, index] = proba
  # Saving the probabilities
  np.save(os.path.join(path_metrics, 'p_dropout.npy'), proba_tab)

#%% Sample

def main(args):
  del args # unused args
  num_samples = 10
  epochs = 10
  batch_size = 32
  p_dropout = 0.5
  thinning_interval = 1
  lr = 0.01
  if FLAGS.dataset == 'mnist':
    launch_mnist_sgd_sgld(epochs, batch_size, thinning_interval, lr)
    launch_mnist_dropout(epochs, batch_size, p_dropout, num_samples)
  else:
    launch_cifar_sgd_sgld(epochs, batch_size, thinning_interval, lr)
    launch_cifar_dropout(epochs, batch_size, p_dropout, num_samples)
    
if __name__ == '__main__':
  app.run(main)
  
# FLAGS.__delattr__('dataset')
  
