# -*- coding: utf-8 -*-
"""Utility functions"""

#%% Imports

import os
import csv
import itertools

import numpy as np

#%% Utility functions

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


def write_to_csv(output_dir, dic):
  with open(os.path.join(output_dir, 'params.csv'), 'w') as csv_file:
      writer = csv.writer(csv_file)
      for key, value in dic.items():
         writer.writerow([key, value])

   
def create_initial_outputs_dir():
  l1 = ['full_network', 'last_layer']
  l2 = ['dropout', 'sgd_sgld', 'bootstrap']
  l3 = ['mnist', 'cifar10', 'cifar100']
  for x in itertools.product(l1, l2, l3):
    path = 'outputs/{}/{}/{}/'.format(x[0], x[1], x[2])
    if not os.path.isdir(path):
      os.makedirs(path)
  if not os.path.isdir('outputs/one_point_estimates'):
    os.makedirs('outputs/one_point_estimates')
  if not os.path.isdir('saved_models'):
    os.makedirs('saved_models')


def create_run_dir(path_dir):
  files = os.listdir(path_dir)
  if not files:
    path = os.path.join(path_dir, 'run_0')
  else:
    path = os.path.join(path_dir, 'run_' + str(len(files)))
  os.makedirs(path)
  return path


def select_classes(y_train, n_class, method='first'):
  """Sample a sub-sample of the training class.
  
  Args:
    y_train: y_train, numpy array (n_test, num_classes), one-hot encoding.
    n_class: int, number of retained classes
    method: string, 'first', the first classes are retained for training,
            'last', the last, 'random', n_class are randomly chosen.
  Returns:
    sec: a selection of indices among n_test corresponding to the selected 
         classes
    index: a boolean vector of length num_classes corresponding to the 
           selected classes
  """
  num_classes = y_train.shape[1]

  if n_class > num_classes:
    raise ValueError('the number of classes to sample from'
                     'is superior to the number of classes of the dataset')
  elif n_class == num_classes:
    print('All the classes are sampled')
    return np.arange(y_train.shape[0]), np.ones(num_classes, dtype=bool)
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
    
  sec = np.dot(y_train, index).astype(bool)  
  return sec, index