"""Experimental purpose."""

#%% Imports

import os
import glob
from absl import app
import pickle

import numpy as np
import keras

import last_layer_algos.last_layer as last_layer
import utils.sgld as sgld
import representation.cifar as cifar
import representation.cifar100 as cifar100
import utils.util as util

#%% Launch LMC

def save_features(dataset):
  if dataset == 'cifar10':
    (x_train, y_train), (x_test, y_test) = cifar.input_cifar10()
    model = cifar.build_model(x_train, 10)
    model_path = 'saved_models/keras_cifar10_trained_model.h5'
  else:
    (x_train, y_train), (x_test, y_test) = cifar.input_cifar100()
    model = cifar100.build_model(x_train, 100)
    model_path = 'saved_models/andrewkruger_cifar100.h5'
  features_train = last_layer.features_extraction(model, model_path, x_train)
  features_test = last_layer.features_extraction(model, model_path, x_test)
  np.save('saved_models/{}_features/features_train.npy'.format(dataset), 
          features_train)
  np.save('saved_models/{}_features/features_test.npy'.format(dataset), 
          features_test)

def launch_cifar_lmc(dataset, epochs, thinning_interval, lr):
  output_dir = util.create_run_dir('outputs/last_layer/'
                                   'lmc/{}'.format(dataset), lr)
  dic_params = {'epochs': epochs,
                'thinning_interval': thinning_interval,
                'lr': lr
               }
  # Save params
  util.write_to_csv(output_dir, dic_params)  
  if dataset == 'cifar10':
    (x_train, y_train), (x_test, y_test) = cifar.input_cifar10()
    model_path = 'saved_models/keras_cifar10_trained_model.h5'
    num_classes = 10
  else:
    (x_train, y_train), (x_test, y_test) = cifar.input_cifar100()
    model_path = 'saved_models/andrewkruger_cifar100.h5'
    num_classes = 100
  
  batch_size = x_train.shape[0]
  
  features_train = np.load('saved_models/{}_features/'
                           'features_train.npy'.format(dataset))
  features_test = np.load('saved_models/{}_features/'
                          'features_test.npy'.format(dataset))
  model = last_layer.build_last_layer(model_path, features_train, num_classes)
  # Create metrics dir
  path_metrics = os.path.join(output_dir, 'metrics')
  os.makedirs(path_metrics)
  # Sampling
  optimizer = sgld.SGLD(x_train.shape[0], lr=lr)
  
  # Compile and train model
  model.compile(optimizer=optimizer, 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
  # Saving after every N batches
  # https://stackoverflow.com/questions/43794995/python-keras-saving-model-weights-after-every-n-batches
  mc = keras.callbacks.ModelCheckpoint(os.path.join(output_dir, 
                                                    'weights{epoch:05d}.h5'),
                                       save_weights_only=True, 
                                       period=thinning_interval)

  hist = model.fit(features_train, y_train,
                   batch_size=batch_size,
                   epochs=epochs,
                   verbose=1,
                   validation_data=(features_test, y_test),
                   callbacks=[mc])
  # Sanity check
  score = model.evaluate(features_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
  
  # Save history of the training
  with open(os.path.join(path_metrics, 
                         'hist_lmc.pkl'), 'wb') as handle:
    pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print('End of sampling.')


def compute_probabilities():
  output_dir = 'outputs/last_layer/lmc/cifar100/run_lr_0.01'
  (x_train, y_train), (x_test, y_test) = cifar.input_cifar100()
  model_path = 'saved_models/andrewkruger_cifar100.h5'
  num_classes = 100
  
  dataset = 'cifar100'

  features_train = np.load('saved_models/{}_features/'
                           'features_train.npy'.format(dataset))
  features_test = np.load('saved_models/{}_features/'
                          'features_test.npy'.format(dataset))
  model = last_layer.build_last_layer(model_path, features_train, num_classes)

  weights_files = [i for i in glob.glob(os.path.join(output_dir, '*.h5'))]
  num_samples = len(weights_files)
  proba_tab = np.zeros(shape=(100, 
                              num_classes, 
                              num_samples))
  for index, weights in enumerate(weights_files):
    model.load_weights(weights)
    proba = model.predict(features_test)
    proba_tab[:, :, index] = proba[:100, :]
      
  path_proba = os.path.join(output_dir, 'metrics', 'p_lmc.npy')
  np.save(path_proba, proba_tab)

  return proba_tab

#%% Sample

def main(argv):
  lr = float(argv[1]) # 0.01
  dataset = 'cifar100'
  epochs = 10000
  thinning_interval = 10
#  save_features(dataset)
#  launch_cifar_lmc(dataset, epochs, thinning_interval, lr)
  compute_probabilities()
    
if __name__ == '__main__':
  app.run(main)  # mnist, cifar10, cifar100
  
  
