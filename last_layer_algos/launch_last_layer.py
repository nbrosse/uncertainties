#%% Imports

import os
from absl import app
import pickle

import numpy as np
import keras

import last_layer_algos.last_layer as last_layer
import utils.sgld as sgld
import representation.mnist as mnist
import representation.cifar as cifar
import representation.cifar100 as cifar100
import utils.util as util

#%% Launch functions

def launch_mnist_sgd_sgld(epochs, batch_size, thinning_interval, lr):
  output_dir = util.create_run_dir('outputs/last_layer/sgd_sgld/mnist')
  dic_params = {'epochs': epochs,
                'batch_size': batch_size,
                'thinning_interval': thinning_interval,
                'lr': lr
               }
  # Save params
  util.write_to_csv(output_dir, dic_params)
  (x_train, y_train), (x_test, y_test) = mnist.input_data()
  model = mnist.build_model(10)
  model_path = 'saved_models/mnist.h5'
  features_train = last_layer.features_extraction(model, model_path, x_train)
  features_test = last_layer.features_extraction(model, model_path, x_test)
  features_shape = (features_train.shape[1],)
  submodel = last_layer.build_last_layer(model_path, features_shape, 10)
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
    hist = last_layer.sgd_sgld_last_layer(submodel, optimizer, epochs, 
                                          batch_size, features_train, 
                                          y_train, features_test, 
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
    proba_tab = last_layer.predict_sgd_sgld_last_layer(submodel, features_test,
                                                       10, path_dic[opt])
    np.save(os.path.join(path_metrics, 'p_{}.npy'.format(opt)), proba_tab)
  print('End of computing probabilities')


def launch_mnist_bootstrap(epochs, batch_size, num_samples, lr):
  output_dir = util.create_run_dir('outputs/last_layer/bootstrap/mnist')
  dic_params = {'epochs': epochs,
                'batch_size': batch_size,
                'num_samples': num_samples,
                'lr': lr
               }
  # Save params
  util.write_to_csv(output_dir, dic_params)
  (x_train, y_train), (x_test, y_test) = mnist.input_data()
  model = mnist.build_model(10)
  model_path = 'saved_models/mnist.h5'
  features_train = last_layer.features_extraction(model, model_path, x_train)
  features_test = last_layer.features_extraction(model, model_path, x_test)
  features_shape = (features_train.shape[1],)
  submodel = last_layer.build_last_layer(model_path, features_shape, 10)
  submodel.compile(optimizer=keras.optimizers.SGD(lr=lr), 
                   loss='categorical_crossentropy', 
                   metrics=['accuracy'])
  # Create metrics dir
  path_metrics = os.path.join(output_dir, 'metrics')
  os.makedirs(path_metrics)
  # Bootstrap
  for i in np.arange(num_samples):
    bootstrap_features_train, bootstrap_y_train = util.bootstrap(
        features_train, y_train)
    submodel = last_layer.bootstrap_last_layer(submodel, epochs, batch_size, 
                                               bootstrap_features_train, 
                                               bootstrap_y_train, 
                                               features_test, y_test, 
                                               model_path)
    # Saving the model
    submodel.save_weights(os.path.join(output_dir, 'weights_{}.h5'.format(i)))
    print('End of boostrap {}'.format(i))
  print('End of sampling.')
  # Compute the probabilities
  proba_tab = np.zeros((features_test.shape[0], y_test.shape[1], num_samples))
  for i in np.arange(num_samples):
    submodel.load_weights(os.path.join(output_dir, 'weights_{}.h5'.format(i)))
    proba_tab[:, :, i] = submodel.predict(features_test)
  # Save proba tab
  np.save(os.path.join(path_metrics, 'p_bootstrap.npy'), proba_tab)
  print('End of computing probabilities')


def launch_mnist_dropout(epochs, batch_size, p_dropout, num_samples, lr):
  output_dir = util.create_run_dir('outputs/last_layer/dropout/mnist')
  dic_params = {'epochs': epochs,
                'batch_size': batch_size,
                'p_dropout': p_dropout,
                'num_samples': num_samples,
                'lr': lr
               }
  # Save params
  util.write_to_csv(output_dir, dic_params)  
  (x_train, y_train), (x_test, y_test) = mnist.input_data()
  model = mnist.build_model(10)
  model_path = 'saved_models/mnist.h5'
  features_train = last_layer.features_extraction(model, model_path, x_train)
  features_test = last_layer.features_extraction(model, model_path, x_test)
  features_shape = (features_train.shape[1],)
  submodel = last_layer.build_last_layer(model_path, features_shape, 10,
                                         p_dropout=p_dropout)
  # Create metrics dir
  path_metrics = os.path.join(output_dir, 'metrics')
  os.makedirs(path_metrics)
  # Training
  submodel.compile(optimizer=keras.optimizers.SGD(lr=lr), 
                   loss='categorical_crossentropy', 
                   metrics=['accuracy'])
  hist = submodel.fit(features_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      validation_data=(features_test, y_test))
  # Saving the model
  submodel.save_weights(os.path.join(output_dir, 'weights.h5'))
  # Save history of the training
  with open(os.path.join(path_metrics, 'hist_dropout.pkl'), 'wb') as handle:
      pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
  # Sampling
  proba_tab = np.zeros(shape=(features_test.shape[0], 10, num_samples))
  for index in np.arange(num_samples):
    proba = submodel.predict(features_test)
    proba_tab[:, :, index] = proba
  # Saving the probabilities
  np.save(os.path.join(path_metrics, 'p_dropout.npy'), proba_tab)

#%% Cifar
  
def launch_cifar_sgd_sgld(dataset, epochs, batch_size, thinning_interval, lr):
  output_dir = util.create_run_dir('outputs/last_layer/'
                                   'sgd_sgld/{}'.format(dataset))
  dic_params = {'epochs': epochs,
                'batch_size': batch_size,
                'thinning_interval': thinning_interval,
                'lr': lr
               }
  # Save params
  util.write_to_csv(output_dir, dic_params)  
  if dataset == 'cifar10':
    (x_train, y_train), (x_test, y_test) = cifar.input_cifar10()
    model = cifar.build_model(x_train, 10)
    model_path = 'saved_models/keras_cifar10_trained_model.h5'
    num_classes = 10
  else:
    (x_train, y_train), (x_test, y_test) = cifar.input_cifar100()
    model = cifar100.build_model(x_train, 100)
    model_path = 'saved_models/andrewkruger_cifar100.h5'
    num_classes = 100
  
  features_train = last_layer.features_extraction(model, model_path, x_train)
  features_test = last_layer.features_extraction(model, model_path, x_test)
  features_shape = (features_train.shape[1],)
  submodel = last_layer.build_last_layer(model_path, features_shape, num_classes)
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
    hist = last_layer.sgd_sgld_last_layer(submodel, optimizer, epochs, 
                                          batch_size, features_train, 
                                          y_train, features_test, 
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
    proba_tab = last_layer.predict_sgd_sgld_last_layer(submodel, features_test,
                                                       num_classes, 
                                                       path_dic[opt])
    np.save(os.path.join(path_metrics, 'p_{}.npy'.format(opt)), proba_tab)
  print('End of computing probabilities')


def launch_cifar_bootstrap(dataset, epochs, batch_size, num_samples, lr):
  output_dir = util.create_run_dir('outputs/last_layer/'
                                   'bootstrap/{}'.format(dataset))
  dic_params = {'epochs': epochs,
                'batch_size': batch_size,
                'num_samples': num_samples,
                'lr': lr
               }
  # Save params
  util.write_to_csv(output_dir, dic_params)
  if dataset == 'cifar10':
    (x_train, y_train), (x_test, y_test) = cifar.input_cifar10()
    model = cifar.build_model(x_train, 10)
    model_path = 'saved_models/keras_cifar10_trained_model.h5'
    num_classes = 10
  else:
    (x_train, y_train), (x_test, y_test) = cifar.input_cifar100()
    model = cifar100.build_model(x_train, 100)
    model_path = 'saved_models/andrewkruger_cifar100.h5'
    num_classes = 100
  features_train = last_layer.features_extraction(model, model_path, x_train)
  features_test = last_layer.features_extraction(model, model_path, x_test)
  features_shape = (features_train.shape[1],)
  submodel = last_layer.build_last_layer(model_path, 
                                         features_shape, num_classes)
  submodel.compile(optimizer=keras.optimizers.SGD(lr=lr), 
                   loss='categorical_crossentropy', 
                   metrics=['accuracy'])
  # Create metrics dir
  path_metrics = os.path.join(output_dir, 'metrics')
  os.makedirs(path_metrics)
  # Bootstrap
  for i in np.arange(num_samples):
    bootstrap_features_train, bootstrap_y_train = util.bootstrap(
        features_train, y_train)
    submodel = last_layer.bootstrap_last_layer(submodel, epochs, batch_size, 
                                               bootstrap_features_train, 
                                               bootstrap_y_train, 
                                               features_test, y_test, 
                                               model_path)
    # Saving the model
    submodel.save_weights(os.path.join(output_dir, 'weights_{}.h5'.format(i)))
    print('End of boostrap {}'.format(i))
  print('End of sampling.')
  # Compute the probabilities
  proba_tab = np.zeros((features_test.shape[0], y_test.shape[1], num_samples))
  for i in np.arange(num_samples):
    submodel.load_weights(os.path.join(output_dir, 'weights_{}.h5'.format(i)))
    proba_tab[:, :, i] = submodel.predict(features_test)
  # Save proba tab
  np.save(os.path.join(path_metrics, 'p_bootstrap.npy'), proba_tab)
  print('End of computing probabilities')
  

def launch_cifar_dropout(dataset, epochs, batch_size, p_dropout, 
                         num_samples, lr):
  output_dir = util.create_run_dir('outputs/last_layer/'
                                   'dropout/{}'.format(dataset))
  dic_params = {'epochs': epochs,
                'batch_size': batch_size,
                'p_dropout': p_dropout,
                'num_samples': num_samples,
                'lr': lr
               }
  # Save params
  util.write_to_csv(output_dir, dic_params)   
  if dataset == 'cifar10':
    (x_train, y_train), (x_test, y_test) = cifar.input_cifar10()
    model = cifar.build_model(x_train, 10)
    model_path = 'saved_models/keras_cifar10_trained_model.h5'
    num_classes = 10
  else:
    (x_train, y_train), (x_test, y_test) = cifar.input_cifar100()
    model = cifar100.build_model(x_train, 100)
    model_path = 'saved_models/andrewkruger_cifar100.h5'
    num_classes = 100
  
  features_train = last_layer.features_extraction(model, model_path, x_train)
  features_test = last_layer.features_extraction(model, model_path, x_test)
  features_shape = (features_train.shape[1],)
  submodel = last_layer.build_last_layer(model_path, features_shape, 
                                         num_classes, p_dropout=p_dropout)
  # Create metrics dir
  path_metrics = os.path.join(output_dir, 'metrics')
  os.makedirs(path_metrics)
  # Training
  submodel.compile(optimizer=keras.optimizers.SGD(lr=lr), 
                   loss='categorical_crossentropy', 
                   metrics=['accuracy'])
  hist = submodel.fit(features_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      validation_data=(features_test, y_test))
  # Saving the model
  submodel.save(os.path.join(output_dir, 'weights.h5'))
  # Save history of the training
  with open(os.path.join(path_metrics, 'hist_dropout.pkl'), 'wb') as handle:
      pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)  
  # Sampling
  proba_tab = np.zeros(shape=(features_test.shape[0], num_classes, 
                              num_samples))
  for index in np.arange(num_samples):
    proba = submodel.predict(features_test)
    proba_tab[:, :, index] = proba
  # Saving the probabilities
  np.save(os.path.join(path_metrics, 'p_dropout.npy'), proba_tab)


#%% Sample

def main(argv):
  dataset = argv[1]  # argv[0] = name of the script file
  num_samples = 10
  epochs = 10
  batch_size = 32
  p_dropout = 0.5
  thinning_interval = 1
  lr = 0.01
  if dataset == 'mnist':
    launch_mnist_sgd_sgld(epochs, batch_size, thinning_interval, lr)
    launch_mnist_dropout(epochs, batch_size, p_dropout, num_samples, lr)
    launch_mnist_bootstrap(epochs, batch_size, num_samples, lr)
  else:
    launch_cifar_sgd_sgld(dataset, epochs, batch_size, thinning_interval, lr)
    launch_cifar_dropout(dataset, epochs, batch_size, p_dropout, 
                         num_samples, lr)
    launch_cifar_bootstrap(dataset, epochs, batch_size, num_samples, lr)
    
if __name__ == '__main__':
  app.run(main)  # mnist, cifar10, cifar100
  
  
