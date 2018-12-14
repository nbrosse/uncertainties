# -*- coding: utf-8 -*-
"""Utility functions"""

#%% Imports

import os
import shutil
import csv

import numpy as np
from utils.dropout_layer import PermaDropout
from keras.layers import Input, Dense
from keras.models import Sequential, Model
import keras

#%% Utility functions

def write_to_csv(output_dir, dic):
  with open(output_dir, 'w') as csv_file:
      writer = csv.writer(csv_file)
      for key, value in dic.items():
         writer.writerow([key, value])

# to decide for p_dropout: if you put None or 0 for non dropout_algo
def create_run_dir(path_dir, hparams):
  data=hparams['dataset']
  algo= hparams['algorithm']
  cl=hparams['num_classes']
  ep=hparams['epochs']
  thInt=hparams['thinning_interval']
  lr=hparams['lr']
  samples=hparams['num_samples']
  p_drop=hparams['p_dropout']

  path_name='{}_{}_cl{}_ep{}_lr{}_samples{}'.format(data,algo,cl,ep,lr,samples)
  if algo=='dropout':
    path_name=path_name+'_pdrop%02d' % p_drop
  path = os.path.join(path_dir, path_name)

  if os.path.isdir(path):
    print('Suppression of old directory with same parameters')
    shutil.rmtree(path, ignore_errors=True)
  os.makedirs(path)
  return path


def cummean(arr, axis):
  """Returns the cumulative mean of array along the axis.

  Args:
    arr: numpy array
    axis: axis over which to compute the cumulative mean.
  """
  n = arr.shape[axis]
  res = np.cumsum(arr, axis=axis)
  res = np.apply_along_axis(lambda x: np.divide(x, np.arange(1, n+1)),
                            axis=axis, arr=res)
  return res


def build_last_layer(features_train, num_classes, 
                     p_dropout=None):
  """Build the last layer keras model.
  
  Args:
    features_train: features of the trainig set.
    num_classes: int, number of classes.
    p_dropout: float between 0 and 1. Fraction of the input units to drop.
  Returns:
    submodel: last layer model.
  """
  n = features_train.shape[0]
  features_shape = (features_train.shape[1],)
  if p_dropout is not None:
    x = Input(shape=features_shape, name='ll_input')
    y = PermaDropout(p_dropout, name='ll_dropout')(x)
    y = Dense(num_classes, activation='softmax', name='ll_dense',
              kernel_regularizer=keras.regularizers.l2(1./n),
              bias_regularizer=keras.regularizers.l2(1./n))(y)
    model = Model(inputs=x, outputs=y)
  else:
    model = Sequential()
    model.add(Dense(num_classes, activation='softmax', 
                    input_shape=features_shape, name='ll_dense',
                    kernel_regularizer=keras.regularizers.l2(1./n),
                    bias_regularizer=keras.regularizers.l2(1./n)))
  return model


def bootstrap(x_train, y_train):
  n = x_train.shape[0]
  index = np.random.choice(n, n, replace=True)
  return x_train[index, :], y_train[index, :]


def select_classes(y_train, n_class, method='first'):
  """Sample a sub-sample of the training class.
  
  Args:
    y_train: y_train, numpy array (n_test, num_classes), one-hot encoding.
    n_class: int, number of retained classes
    method: string, 'first', the first classes are retained for training,
            'last', the last, 'random', n_class are randomly chosen.
  Returns:
    index: a boolean vector of length num_classes corresponding to the 
           selected classes
  """
  num_classes = y_train.shape[1]

  if n_class > num_classes:
    raise ValueError('the number of classes to sample from'
                     'is superior to the number of classes of the dataset')
  elif n_class == num_classes:
    print('All the classes are sampled')
    return np.ones(num_classes, dtype=bool)
  else:
    print('{} classes out of {} classes ' 
          'are sampled.'.format(n_class, num_classes))
  
  index = np.zeros(num_classes, dtype=bool)

  if method == 'first':
    index[:n_class] = 1
  elif method == 'last':
    index[-n_class:] = 1
  else:
    index[np.random.choice(num_classes, size=n_class, replace=False)] = 1
    
  return index

