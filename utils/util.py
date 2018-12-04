# -*- coding: utf-8 -*-
"""Utility functions"""

#%% Imports

import os
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

def create_initial_outputs_dir():
  l1 = ['full_network', 'last_layer']
  l2 = ['dropout', 'sgd_sgld']
  l3 = ['mnist', 'cifar10', 'cifar100']
  for x in itertools.product(l1, l2, l3):
    path = 'outputs/{}/{}/{}/'.format(x[0], x[1], x[2])
    if not os.isdir(path):
      os.makedirs(path)
  if not os.isdir('saved_models'):
    os.makedirs('saved_models')

def create_run_dir(path_dir):
  files = os.listdir(path_dir)
  if not files:
    path = os.path.join(path_dir, 'run_0')
  else:
    path = os.path.join(path_dir, 'run_' + str(len(files)))
  os.makedirs(path)
  return path

def class_sampling(x_train, y_train, n_class, method='first'):
  """Sample a sub-sample of the training class.
  
  Args:
    x_train: x_train
    y_train: y_train, numpy array (n_test, num_classes), one-hot encoding.
    n_class: int, number of retained classes
    method: string, 'first', the first classes are retained for training,
            'last', the last, 'random', n_class are randomly chosen.
  Returns:
  """
  num_classes = y_train[0])
  classes_sampled=list(np.zeros(num_classes))

  if num_classes > total_classes:
      raise ValueError('the number of classes to sample is superior to the number of classes of the dataset')
  if num_classes==total_classes:
      print('all the classes are sampled')

  selected_index=[]
  if classes==None:
      if method=='first':
          classes_sampled=list(range(0,num_classes))
          for index in range(y_train.shape[0]):
              if np.sum(y_train[index][:num_classes])==1:
                  selected_index.append(index)
      elif method=='last':
          classes_sampled=list(range(total_classes-num_classes, total_classes))
          for index in range(y_train.shape[0]):
              if np.sum(y_train[index][-num_classes:])==1:
                  selected_index.append(index)
      elif method=='random':
          classes_sampled=list(np.random.choice(range(total_classes), num_classes, replace=False))
          for index in range(y_train.shape[0]):
              class_values=[y_train[index][cl] for cl in classes_sampled]
              if np.sum(class_values)==1:
                  selected_index.append(index)
  else:
      classes_sampled=classes
      for index in range(y_train.shape[0]):
          class_values=[y_train[index][cl] for cl in classes_sampled]
          if np.sum(class_values)==1:
              selected_index.append(index)

  x_train_restricted=x_train[selected_index]
  y_train_restricted=y_train[selected_index]
  # re_index the sampled datasets?
  oos_classes=[cl for cl in list(range(total_classes)) if not cl in classes_sampled]

  return x_train_restricted, y_train_restricted, classes_sampled, oos_classes