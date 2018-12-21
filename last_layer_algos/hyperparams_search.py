""" Hyperparameters search for the last layer algorithms.

The output_dir should contain experiments relative to one dataset 
and one algorithm.
"""


#%% Imports

#  from __future__ import absolute_import
#  from __future__ import division
#  from __future__ import print_function

# https://stackoverflow.com/questions/21485319/high-memory-usage-using-python-multiprocessing

import os
import h5py
import glob
from psutil import virtual_memory

import numpy as np
import pandas as pd

import utils.metrics as metrics

#import multiprocessing as mp  
# Not working on windows currently.
# Problems of Out Of Memory errors, even on linux not solved.
# In that case, a sequential implementation has to be executed.

#nb_cores = mp.cpu_count()
mem = virtual_memory()
MEM = mem.total / 4.0 # (2.0 * nb_cores) # max physical memory for us


#%% Compute AURC

def compute_aurc(y, h5file):
  """Compute AURC.
  
  Args:
    y: true y, one-hot encoding size (n_test, n_class)
    h5file: path to the h5 file containing the tab of probabilities.
  Returns:
    aurc: dict such that 
          aurc['std'] contains aurc associated to the confidence function -std of p
          aurc['softmax'], max of p
          aurc['q'], entropy of q and q as classifier.
  """
  memsize = os.path.getsize(h5file)
  nb_chunks = int(memsize // MEM + 1) 
  f = h5py.File(os.path.join(h5file), 'r')  # read mode
  h5data = f['proba']
  num_items_per_chunk = h5data.shape[0] // nb_chunks 
  # number of chunks is equal to nb_chunks or nb_chunks + 1
  p_mean, p_std, q_tab = [], [], []
  for i in np.arange(nb_chunks + 1):
    if i < nb_chunks:
      p_i = h5data[i*num_items_per_chunk:(i+1)*num_items_per_chunk, :, :]
      y_i = y[i*num_items_per_chunk:(i+1)*num_items_per_chunk, :]
    elif h5data.shape[0] % nb_chunks == 0:  # i == nb_chunks
      # number of chunks is equal to nb_chunks
      break
    else:
      # number of chunks is equal to nb_chunks + 1
      p_i = h5data[i*num_items_per_chunk:, :, :]
      y_i = y[i*num_items_per_chunk:, :]
    res_dic = metrics.compute_metrics(y_i, p_i)
    p_mean.append(res_dic['p_mean'])
    p_std.append(res_dic['p_std'])
    q_tab.append(res_dic['q_tab'])
  p_mean = np.vstack(tuple(p_mean))
  p_std = np.vstack(tuple(p_std))
  q_tab = np.vstack(tuple(q_tab))
  res_dic = metrics.aurc(y, p_mean, p_std, q_tab)
  aurc = {}
  aurc['std'] = res_dic['risk_cov_std']['aurc']
  aurc['softmax'] = res_dic['risk_cov_softmax']['aurc']
  aurc['q'] = res_dic['risk_cov_q']['aurc']
  f.close()
  return aurc

#%% Hyperparameters search

dataset = 'cifar100-first-100'
output_dir = 'outputs/last_layer/{}_*'.format(dataset)
npzfile = np.load('saved_models/{}/y.npz'.format(dataset))
y = npzfile['y_test_in']
list_experiments = glob.glob(output_dir)

# Windows
if os.name == 'nt':
  temp = []
  for experiment in list_experiments:
    temp.append(experiment.replace('\\', '/'))
  list_experiments = temp

def launch_aurc(experiment):
  """Launch the computation of aurc for one experiment."""
  # Create dict to store params and aurc => to pd.DataFrame
  dic = {'dataset': [],
         'algorithm': [],
         'aurc_std': [],
         'aurc_softmax': [],
         'aurc_q': []
        }
  list_params = experiment.split('/')[-1].split('_')[2:]
  for p in list_params:
    p = p.split('-')[0]
    dic[p] = []
  
  # Launch AURC computation.
  exp = experiment.split('/')[-1]
  exp_split= exp.split('_')
  dataset = exp_split[0]
  algorithm = exp_split[1]
  if algorithm == 'sgdsgld':
    aurc = {}
    aurc['sgd'] = compute_aurc(y, os.path.join(experiment, 'p_sgd_in.h5'))
    aurc['sgld'] = compute_aurc(y, os.path.join(experiment, 'p_sgld_in.h5'))
    for opt in ['sgd', 'sgld']:
      dic['dataset'].append(dataset)
      dic['algorithm'].append(opt)
      for p in exp_split[2:]:
        if len(p.split('-')) == 2:
          p1, p2 = p.split('-') 
          dic[p1].append(float(p2))
        else:
          p1, p2, p3 = p.split('-')
          dic[p1].append(float('-'.join([p2, p3])))
      dic['aurc_std'].append(aurc[opt]['std'])
      dic['aurc_softmax'].append(aurc[opt]['softmax'])
      dic['aurc_q'].append(aurc[opt]['q'])
  else:
    aurc = compute_aurc(y, os.path.join(experiment, 'p_in.h5'))
    dic['dataset'].append(dataset)
    dic['algorithm'].append(algorithm)
    for p in exp_split[2:]:
      if len(p.split('-')) == 2:
        p1, p2 = p.split('-') 
        dic[p1].append(float(p2))
      else:
        p1, p2, p3 = p.split('-')
        dic[p1].append(float('-'.join([p2, p3])))
    dic['aurc_std'].append(aurc['std']) 
    dic['aurc_softmax'].append(aurc['softmax'])
    dic['aurc_q'].append(aurc['q'])
  
  # to dataframe
  df = pd.DataFrame(dic)
  return df
  

# Sequential implementation

res = []
i = 0
for experiment in list_experiments:
  print('---------------')
  print('Step {}'.format(i))
  print('---------------')
  
  df = launch_aurc(experiment)
  res.append(df)
  i += 1

print('Postprocessing')  
df = pd.concat(res, ignore_index=True)
data_algo = list_experiments[0].split('/')[-1].split('_')[:2]
df.to_pickle(os.path.join('outputs/last_layer', '_'.join(data_algo)) + '.pkl')  
print('End')

#if __name__ == '__main__':
#  
#  if os.name == 'nt':
#    print('Multiprocessing not supported on Windows.')
#    df = launch_aurc(list_experiments[0])
#  else:
#    nb_cores = mp.cpu_count()
#    pool = mp.pool.Pool(processes=nb_cores, maxtasksperchild=1)
#    res = pool.map(launch_aurc, list_experiments)
#    pool.close()
#    df = pd.concat(res, ignore_index=True)
#    data_algo = list_experiments[0].split('/')[-1].split('_')[:2]
#    df.to_pickle(os.path.join('outputs/last_layer', '_'.join(data_algo)) + '.pkl')

#%% Using pandas dataframe to select best hyperparams.

#p1 = 'outputs/{}_dropout.pkl'.format(dataset)
#p2 = 'outputs/{}_bootstrap.pkl'.format(dataset)
#p3 = 'outputs/{}_sgdsgld.pkl'.format(dataset)
#
#def process_df(path_to_df):
#  df = pd.read_pickle(path_to_df)
#  df['min_aurc'] = df[['aurc_std', 'aurc_softmax', 'aurc_q']].min(axis=1)
#  df['rel_increase'] = df['min_aurc'].divide(df['min_aurc'].min())
#  df.to_pickle(path_to_df)
#  return df
#  
#df1 = pd.read_pickle(p1)
#df2 = pd.read_pickle(p2)
#df1 = process_df(p1)
#df3 = pd.read_pickle(p3)
