""" Hyperparameters search.
"""


#%% Imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import glob
from psutil import virtual_memory

import numpy as np
import pandas as pd
import pickle

import utils.metrics as metrics

#nb_cores = mp.cpu_count()
mem = virtual_memory()
MEM = mem.total / 10.0 # (2.0 * nb_cores) # max physical memory for us


#%% Compute AURC and ECE

def compute_aurc(y, h5file):
  """Compute AURC from a h5file.
  
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

def compute_ece(y, h5file):
  """Compute ECE from a h5file.
  
  Args:
    y: true y, one-hot encoding size (n_test, n_class)
    h5file: path to the h5 file containing the tab of probabilities.
  Returns:
    ece: expected calibration error.
  """
  memsize = os.path.getsize(h5file)
  nb_chunks = int(memsize // MEM + 1) 
  f = h5py.File(os.path.join(h5file), 'r')  # read mode
  h5data = f['proba']
  num_items_per_chunk = h5data.shape[0] // nb_chunks 
  # number of chunks is equal to nb_chunks or nb_chunks + 1
  p_mean = []
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
  p_mean = np.vstack(tuple(p_mean))
  cal = metrics.calibration(y, p_mean)
  ece = cal['ece']
  f.close()
  return ece


#%% Launch AURC, ECE, AUROC and AUPR for one experiment.


def launch_aurc(y, experiment):
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
  df['min_aurc'] = df[['aurc_std', 'aurc_softmax', 'aurc_q']].min(axis=1)
  
  return df


def launch_ece(y, experiment):
  """Launch the computation of ece for one experiment."""
  # Create dict to store params and ece => to pd.DataFrame
  dic = {'dataset': [],
         'algorithm': [],
         'ece': []
        }
  list_params = experiment.split('/')[-1].split('_')[2:]
  for p in list_params:
    p = p.split('-')[0]
    dic[p] = []
  
  # Launch ECE computation.
  exp = experiment.split('/')[-1]
  exp_split = exp.split('_')
  dataset = exp_split[0]
  algorithm = exp_split[1]
  if algorithm == 'sgdsgld':
    ece = {}
    ece['sgd'] = compute_ece(y, os.path.join(experiment, 'p_sgd_in.h5'))
    ece['sgld'] = compute_ece(y, os.path.join(experiment, 'p_sgld_in.h5'))
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
      dic['ece'].append(ece[opt])
  else:
    ece = compute_ece(y, os.path.join(experiment, 'p_in.h5'))
    dic['dataset'].append(dataset)
    dic['algorithm'].append(algorithm)
    for p in exp_split[2:]:
      if len(p.split('-')) == 2:
        p1, p2 = p.split('-') 
        dic[p1].append(float(p2))
      else:
        p1, p2, p3 = p.split('-')
        dic[p1].append(float('-'.join([p2, p3])))
    dic['ece'].append(ece) 
  
  # to dataframe
  df = pd.DataFrame(dic)

  return df


def launch_auroc_aupr(experiment):
  """Launch the computation of AUROC and AUPR for one experiment."""
  # Create dict to store params and ece => to pd.DataFrame
  dic = {'dataset': [],
         'algorithm': [],
         'auroc_std': [],
         'auroc_maxmin': [],
         'auroc_q': [],
         'auroc_softmax': [],
         'aupr_in_softmax': [],
         'aupr_in_maxmin': [],
         'aupr_in_std': [],
         'aupr_in_q': [],
         'aupr_out_softmax': [],
         'aupr_out_maxmin': [],
         'aupr_out_std': [],
         'aupr_out_q': []
        }
  list_params = experiment.split('/')[-1].split('_')[2:]
  for p in list_params:
    p = p.split('-')[0]
    dic[p] = []
  
  exp = experiment.split('/')[-1]
  exp_split = exp.split('_')
  dataset = exp_split[0]
  algorithm = exp_split[1]
  if algorithm == 'sgdsgld':
    res = {}
    res['sgd'] = metrics.metrics_ood_from_h5files(
        os.path.join(experiment, 'p_sgd_in.h5'), 
        os.path.join(experiment, 'p_sgd_out.h5'))
    res['sgld'] = metrics.metrics_ood_from_h5files(
         os.path.join(experiment, 'p_sgld_in.h5'), 
         os.path.join(experiment, 'p_sgld_out.h5'))    
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
      dic['auroc_std'].append(res[opt]['auroc_aupr_std']['auroc'])
      dic['auroc_softmax'].append(res[opt]['auroc_aupr_softmax']['auroc'])
      dic['auroc_maxmin'].append(res[opt]['auroc_aupr_maxmin']['auroc'])
      dic['auroc_q'].append(res[opt]['auroc_aupr_q']['auroc'])
      dic['aupr_in_std'].append(res[opt]['auroc_aupr_std']['aupr_in'])
      dic['aupr_in_softmax'].append(res[opt]['auroc_aupr_softmax']['aupr_in'])
      dic['aupr_in_maxmin'].append(res[opt]['auroc_aupr_maxmin']['aupr_in'])
      dic['aupr_in_q'].append(res[opt]['auroc_aupr_q']['aupr_in'])
      dic['aupr_out_std'].append(res[opt]['auroc_aupr_std']['aupr_out'])
      dic['aupr_out_softmax'].append(res[opt]['auroc_aupr_softmax']['aupr_out'])
      dic['aupr_out_maxmin'].append(res[opt]['auroc_aupr_maxmin']['aupr_out'])
      dic['aupr_out_q'].append(res[opt]['auroc_aupr_q']['aupr_out'])
  else:
    res = metrics.metrics_ood_from_h5files(
        os.path.join(experiment, 'p_in.h5'), 
        os.path.join(experiment, 'p_out.h5'))
    dic['dataset'].append(dataset)
    dic['algorithm'].append(algorithm)
    for p in exp_split[2:]:
      if len(p.split('-')) == 2:
        p1, p2 = p.split('-') 
        dic[p1].append(float(p2))
      else:
        p1, p2, p3 = p.split('-')
        dic[p1].append(float('-'.join([p2, p3])))
    dic['auroc_std'].append(res['auroc_aupr_std']['auroc'])
    dic['auroc_softmax'].append(res['auroc_aupr_softmax']['auroc'])
    dic['auroc_maxmin'].append(res['auroc_aupr_maxmin']['auroc'])
    dic['auroc_q'].append(res['auroc_aupr_q']['auroc'])
    dic['aupr_in_std'].append(res['auroc_aupr_std']['aupr_in'])
    dic['aupr_in_softmax'].append(res['auroc_aupr_softmax']['aupr_in'])
    dic['aupr_in_maxmin'].append(res['auroc_aupr_maxmin']['aupr_in'])
    dic['aupr_in_q'].append(res['auroc_aupr_q']['aupr_in'])
    dic['aupr_out_std'].append(res['auroc_aupr_std']['aupr_out'])
    dic['aupr_out_softmax'].append(res['auroc_aupr_softmax']['aupr_out'])
    dic['aupr_out_maxmin'].append(res['auroc_aupr_maxmin']['aupr_out'])
    dic['aupr_out_q'].append(res['auroc_aupr_q']['aupr_out'])
  
  # to dataframe
  df = pd.DataFrame(dic)
  
  df['max_auroc'] = df[['auroc_std', 'auroc_softmax', 
                        'auroc_q', 'auroc_maxmin']].max(axis=1)
  df['max_aupr_in'] = df[['aupr_in_std', 'aupr_in_softmax', 
                          'aupr_in_q', 'aupr_in_maxmin']].max(axis=1)
  df['max_aupr_out'] = df[['aupr_out_std', 'aupr_out_softmax', 
                           'aupr_out_q', 'aupr_out_maxmin']].max(axis=1)
  
  return df


#%% Launch the computation of the metrics
  
def hyperparams_computation():
  dataset = 'cifar100-first-50'
  output_dir = 'outputs/full_network/{}_*'.format(dataset)
  
  if dataset.split('-')[0] == 'imagenet':
    y = np.load('saved_models/{}/y.npy'.format(dataset))
  else:
    npzfile = np.load('saved_models/{}/y.npz'.format(dataset))
    y = npzfile['y_test_in']
  list_experiments = glob.glob(output_dir)
  
  # Windows
  if os.name == 'nt':
    temp = []
    for experiment in list_experiments:
      temp.append(experiment.replace('\\', '/'))
    list_experiments = temp
  
  
  score_computed = 'AUROC-AUPR' # 'AUROC-AUPR', 'ECE', 'AURC'
  
  print('---------------')
  print('Score computed: {}'.format(score_computed))
  print('---------------')
  
  res = []
  i = 0
  for experiment in list_experiments:
    print('---------------')
    print('Step {}'.format(i))
    print('---------------')
    
    if score_computed == 'AURC':
      df = launch_aurc(y, experiment)
    elif score_computed == 'ECE':
      df = launch_ece(y, experiment)
    elif score_computed == 'AUROC-AUPR':
      df = launch_auroc_aupr(experiment)
    else:
      raise ValueError('this quantity can not be computed.')
    res.append(df)
    i += 1
  
  def _f_ece(group):
    group['increase_ece'] = group['ece'].divide(group['ece'].min())
    return group
  
  def _f_auroc_aupr(group):
    group['increase_auroc'] = group['max_auroc'].divide(group['max_auroc'].max())
    group['increase_aupr_in'] = group['max_aupr_in'].divide(group['max_aupr_in'].max())
    group['increase_aupr_out'] = group['max_aupr_out'].divide(group['max_aupr_out'].max())
    return group
  
  def _f_aurc(group):
    group['increase_aurc'] = group['min_aurc'].divide(group['min_aurc'].min())
    return group


  print('Postprocessing')  
  df = pd.concat(res, ignore_index=True, sort=True)
  if score_computed == 'AURC':
    df = df.groupby('algorithm').apply(_f_aurc)
  elif score_computed == 'ECE':
    df = df.groupby('algorithm').apply(_f_ece)
  elif score_computed == 'AUROC-AUPR':
    df = df.groupby('algorithm').apply(_f_auroc_aupr)
  else:
    raise ValueError('this quantity can not be computed.')
#  data_algo = list_experiments[0].split('/')[-1].split('_')[:2]
  data_algo = list_experiments[0].split('/')[-1].split('_')[0]
  df.to_pickle(os.path.join('outputs/full_network', '_'.join(data_algo)) + '_fullnetwork.pkl')  
  print('End')
  

#hyperparams_computation()

#%% Postprocessing the hyperparams search

def postprocess_algos(dataset):
#  dataset = 'mnist-first-5'
  path_dropout = 'outputs/{}_hyperparams/{}_dropout.pkl'.format(dataset, dataset)
  path_bootstrap = 'outputs/{}_hyperparams/{}_bootstrap.pkl'.format(dataset, dataset)
  path_sgdsgld = 'outputs/{}_hyperparams/{}_sgdsgld.pkl'.format(dataset, dataset)

  df_d = pd.read_pickle(path_dropout)
  df_b = pd.read_pickle(path_bootstrap)
  df_s = pd.read_pickle(path_sgdsgld)
  
  list_df = [df_d, df_b, df_s]
  df = pd.concat(list_df, ignore_index=True, sort=True)
  df_sec = df.loc[df['increase_auroc'] == 1, :].copy()
  
  df.to_pickle('outputs/{}_hyperparams/{}_hparams.pkl'.format(dataset, dataset))
  df_sec.to_pickle('outputs/{}_hyperparams/{}_hparams_sec.pkl'.format(dataset, dataset))
  
  return df, df_sec


def postprocess_dfs_aurc():
  
  dfs = {}
  
  for dataset in ['mnist-first-10', 'cifar10-first-10', 'cifar100-first-100']:
    path_ll = 'outputs/{}_hyperparams/{}_hparams.pkl'.format(dataset, dataset)
    path_full = 'outputs/{}_hyperparams/{}_hparams_fullnetwork.pkl'.format(dataset, dataset)
    
    with open('outputs/last_layer/{}_onepoint/'
              'metrics_dic.pkl'.format(dataset), 'rb') as f:
      onepoint = pickle.load(f)
    
    dic_onepoint = {'algorithm': ['onepoint'],
                    'dataset': [dataset],
                    'aurc_softmax': [onepoint['risk_cov_softmax']['aurc']],
                    'min_aurc': [onepoint['risk_cov_softmax']['aurc']],
                    'ece': [onepoint['cal']['ece']],
                    }
    df_onepoint = pd.DataFrame.from_dict(dic_onepoint)
    
    df_ll = pd.read_pickle(path_ll)
    df_full = pd.read_pickle(path_full)
    df_full['algorithm'] = df_full['algorithm'] + '_full'
    df = df_ll.loc[df_ll['increase_aurc'] == 1, :].copy()
    df_full_sec = df_full.loc[df_full['increase_aurc'] == 1, :].copy()
    df = df.append(df_onepoint, sort=True)
    df = df.append(df_full_sec, sort=True)
    df['increase_aurc'] = df['min_aurc'].divide(df['min_aurc'].min())
    dfs[dataset] = df
    # Compressed view
#    dfs[dataset] = dfs[dataset][['algorithm', 'lr', 's', 'min_aurc', 
#       'increase_aurc', 'ece', 'increase_ece', 'ep', 'pdrop']]
    
  df_total = pd.concat([dfs[dataset] for dataset in ['mnist-first-10', 'cifar10-first-10', 'cifar100-first-100']], 
                       axis=0, ignore_index=True)
  df_total.to_csv('outputs/aurc.csv')
  
  return df_total

def postprocess_dfs_auroc_aupr():
  
  dfs = {}
  list_datasets = ['mnist-first-5', 'cifar10-first-5', 'cifar100-first-50']
  
  for dataset in list_datasets:
    path_ll = 'outputs/{}_hyperparams/{}_hparams.pkl'.format(dataset, dataset)
    path_full = 'outputs/{}_hyperparams/{}_hparams_fullnetwork.pkl'.format(dataset, dataset)
    
    with open('outputs/last_layer/{}_onepoint/'
              'metrics_dic.pkl'.format(dataset), 'rb') as f:
      onepoint = pickle.load(f)
    
    dic_onepoint = {'algorithm': ['onepoint'],
                    'dataset': [dataset],
                    'auroc_softmax': [onepoint['auroc_aupr_softmax']['auroc']],
                    'aupr_in_softmax': [onepoint['auroc_aupr_softmax']['aupr_in']],
                    'aupr_out_softmax': [onepoint['auroc_aupr_softmax']['aupr_out']],
                    'max_auroc': [onepoint['auroc_aupr_softmax']['auroc']],
                    'max_aupr_in': [onepoint['auroc_aupr_softmax']['aupr_in']],
                    'max_aupr_out': [onepoint['auroc_aupr_softmax']['aupr_out']],
                    }
    df_onepoint = pd.DataFrame.from_dict(dic_onepoint)
    
    df_ll = pd.read_pickle(path_ll)
    df_full = pd.read_pickle(path_full)
    df_full['algorithm'] = df_full['algorithm'] + '_full'
    df = df_ll.loc[df_ll['increase_auroc'] == 1, :].copy()
    df_full_sec = df_full.loc[df_full['increase_auroc'] == 1, :].copy()
    df = df.append(df_onepoint, sort=True)
    df = df.append(df_full_sec, sort=True)
    df['increase_auroc'] = df['max_auroc'].divide(df['max_auroc'].max())
    df['increase_aupr_in'] = df['max_aupr_in'].divide(df['max_aupr_in'].max())
    df['increase_aupr_out'] = df['max_aupr_out'].divide(df['max_aupr_out'].max())
    dfs[dataset] = df
    
    # Compressed view
#    dfs[dataset] = dfs[dataset][['algorithm', 'lr', 's', 'min_aurc', 
#       'increase_aurc', 'ece', 'increase_ece', 'ep', 'pdrop']]
    
  df_total = pd.concat([dfs[dataset] for dataset in list_datasets], axis=0)
  df_total.to_csv('outputs/auroc_aupr.csv')
  
  return df_total

