# -*- coding: utf-8 -*-
"""Utility functions"""

#%% Imports

import os
import shutil
import csv

import numpy as np

#%% Utility functions

def write_to_csv(output_dir, dic):
  """Write a python dic to csv."""
  with open(output_dir, 'w') as csv_file:
      writer = csv.writer(csv_file)
      for key, value in dic.items():
         writer.writerow([key, value])


def create_run_dir(path_dir, hparams):
  dataset = hparams['dataset']
  algorithm = hparams['algorithm']
  epochs = hparams['epochs']
  samples = hparams['samples']
  lr = hparams['lr']
  bs = hparams['batch_size']
  p_dropout = hparams['p_dropout']
  
  if algorithm == 'sgdsgld':
    path_name = '{}_{}_lr-{}_bs-{}_s-{}'.format(dataset, algorithm, 
                 lr, bs, samples)
  elif algorithm == 'onepoint':
    path_name = '{}_{}'.format(dataset, algorithm)
  elif algorithm == 'bootstrap':
    path_name = '{}_{}_ep-{}_lr-{}_bs-{}_s-{}'.format(dataset, algorithm, epochs, 
                 lr, bs, samples)
  elif algorithm == 'dropout':
    path_name = '{}_{}_ep-{}_lr-{}_bs-{}_s-{}'.format(dataset, algorithm, epochs, 
                 lr, bs, samples)
    path_name = path_name + '_pdrop-{}'.format(p_dropout)
  else:
    raise ValueError('This algorithm is not supported')
  
  path = os.path.join(path_dir, path_name)

  if os.path.isdir(path):
    print('Suppression of old directory with same parameters')
    os.chmod(path, 0o777)
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

