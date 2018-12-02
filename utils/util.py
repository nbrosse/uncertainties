# -*- coding: utf-8 -*-
"""Utility functions"""

#%% Imports

import os
import itertools

#%% Utility functions

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

