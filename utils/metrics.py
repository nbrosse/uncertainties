"""Metrics for classification.

Compute:
- Brier score
- Negative loglikelihood
- entropy
- accuracy
- mutual information
- q probability and its entropy
- calibration (reliability diagram, maximum calibration error,
               expected calibration error)
- mean of the output predicted probabilities
- std of the output predicted probabilities.
- (max - min) of the output predicted probabilities.
- AURC
- AUROC and AUPR int/out
"""

#%% Imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
from absl import app
import os
import h5py
import gc
#from psutil import virtual_memory

import numpy as np
import scipy.stats as spstats
import utils.util as util

#mem = virtual_memory()
#MEM = mem.total / 40.0 # max physical memory for us

#%% Functions for metrics

def metrics_from_h5file(y, h5file, nb_chunks=1):
  """Compute in-distribution metrics from a (potentially very big) h5file.
  
  Args:
    y: true y, one-hot encoding size (n_test, n_class)
    h5file: path to the h5 file containing the tab of probabilities.
    nb_chunks: int, number of chunks to divide the h5file if it is too big to 
               read in one piece.
  Returns:
    result_dic: dict that contains the metrics.
  """
  f = h5py.File(os.path.join(h5file), 'r')  # read mode
  h5data = f['proba']
  n_test, n_class, n_samples = h5data.shape
  num_items_per_chunk = h5data.shape[0] // nb_chunks 
  # number of chunks is equal to nb_chunks or nb_chunks + 1
  if h5data.shape[0] % nb_chunks == 0:
    num_chunks = nb_chunks # number of chunks is equal to nb_chunks
  else:
    num_chunks = nb_chunks + 1 # number of chunks is equal to nb_chunks + 1
  print('----------------')
  print('Reading file {} divided in {} chunks.'.format(h5file, num_chunks))
  print('----------------')
  p_mean, p_std, q_tab, p_max_min = [], [], [], []
  acc = np.zeros((num_chunks, n_samples))
  bs = np.zeros((num_chunks, n_samples))
  neglog = np.zeros((num_chunks, n_samples))
  ent, ent_q, mi = [], [], []
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
    res_dic = compute_metrics(y_i, p_i)
    p_mean.append(res_dic['p_mean'])
    p_max_min.append(res_dic['p_max_min'])
    p_std.append(res_dic['p_std'])
    q_tab.append(res_dic['q_tab'])
    acc[i, :] = res_dic['acc']
    bs[i, :] = res_dic['bs']
    neglog[i, :] = res_dic['neglog']
    ent.append(res_dic['ent'])
    ent_q.append(res_dic['ent_q'])
    mi.append(res_dic['mi'])
    
    # Clear memory
    del p_i, y_i
    
  acc = np.mean(acc, axis=0)
  bs = np.mean(bs, axis=0)
  neglog = np.mean(neglog, axis=0)

  p_mean = np.vstack(tuple(p_mean))
  p_max_min = np.vstack(tuple(p_max_min))
  p_std = np.vstack(tuple(p_std))
  q_tab = np.vstack(tuple(q_tab))
  ent = np.concatenate(tuple(ent))
  ent_q = np.concatenate(tuple(ent_q))
  mi = np.concatenate(tuple(mi))
  
  # Clear memory
  gc.collect()
  
  cal = calibration(y, p_mean)
  
  risk_cov = aurc(y, p_mean, p_std, q_tab, p_max_min)
  
  result_dic = {}
  
  result_dic = {'acc': acc,  # n_samples
              'bs': bs,  # n_samples
              'p_mean': p_mean,  # (n_test, n_class)
              'p_max_min': p_max_min,  # (n_test, n_class)
              'p_std': p_std,  # (n_test, n_class)
              'neglog': neglog,  # n_samples
              'ent': ent,  # (n_test, n_samples)
              'cal': cal,  # reliability_diag, ece, mce
              'q_tab': q_tab,  # (n_test, n_class) 
              'ent_q': ent_q,  # n_test
              'mi': mi,  # n_test
              'risk_cov_std': risk_cov['risk_cov_std'], # conf, risk_cov, aurc, eaurc
              'risk_cov_softmax': risk_cov['risk_cov_softmax'],
              'risk_cov_maxmin': risk_cov['risk_cov_maxmin'],
              'risk_cov_q': risk_cov['risk_cov_q']
              }
  
  f.close()
  
  # Clear memory 
  del h5data
  gc.collect()
  
  return result_dic


def aurc_from_h5file(y, h5file, nb_chunks=1):
  """Compute AURC from a (potentially very big) h5file.
  
  Args:
    y: true y, one-hot encoding size (n_test, n_class)
    h5file: path to the h5 file containing the tab of probabilities.
    nb_chunks: int, number of chunks to divide the h5file if it is too big to 
               read in one piece.
  Returns:
    result_dic: dict that contains the metrics.
  """
  f = h5py.File(os.path.join(h5file), 'r')  # read mode
  h5data = f['proba']
  n_test, n_class, n_samples = h5data.shape
  num_items_per_chunk = h5data.shape[0] // nb_chunks 
  # number of chunks is equal to nb_chunks or nb_chunks + 1
  if h5data.shape[0] % nb_chunks == 0:
    num_chunks = nb_chunks # number of chunks is equal to nb_chunks
  else:
    num_chunks = nb_chunks + 1 # number of chunks is equal to nb_chunks + 1
  print('----------------')
  print('Reading file {} divided in {} chunks.'.format(h5file, num_chunks))
  print('----------------')
  p_mean, p_std, q_tab, p_max_min = [], [], [], []
  for i in np.arange(nb_chunks + 1):
    if i < nb_chunks:
      p_i = h5data[i*num_items_per_chunk:(i+1)*num_items_per_chunk, :, :]
    elif h5data.shape[0] % nb_chunks == 0:  # i == nb_chunks
      # number of chunks is equal to nb_chunks
      break
    else:
      # number of chunks is equal to nb_chunks + 1
      p_i = h5data[i*num_items_per_chunk:, :, :]
    p_mean.append(np.mean(p_i, axis=2))
    p_max_min.append(np.max(p_i, axis=2) - np.min(p_i, axis=2))
    p_std.append(np.std(p_i, axis=2))
    q_tab.append(q_probability(p_i))
    
    # Clear memory
    del p_i
    
  p_mean = np.vstack(tuple(p_mean))
  p_max_min = np.vstack(tuple(p_max_min))
  p_std = np.vstack(tuple(p_std))
  q_tab = np.vstack(tuple(q_tab))
  
  # Clear memory
  gc.collect()
  
  risk_cov = aurc(y, p_mean, p_std, q_tab, p_max_min)
  
  dic_aurc = {}
  
  for method in risk_cov.keys():
    dic_aurc[method.split('_')[-1]] = risk_cov[method]['aurc']
  
  # Clear memory
  del p_mean, p_std, q_tab, p_max_min
  del risk_cov
  
  f.close()
  
  # Clear memory
  del h5data
  gc.collect()
  
  return dic_aurc


def ece_from_h5file(y, h5file, nb_chunks=1):
  """Compute AURC from a (potentially very big) h5file.
  
  Args:
    y: true y, one-hot encoding size (n_test, n_class)
    h5file: path to the h5 file containing the tab of probabilities.
    nb_chunks: int, number of chunks to divide the h5file if it is too big to 
               read in one piece.
  Returns:
    result_dic: dict that contains the metrics.
  """
  f = h5py.File(os.path.join(h5file), 'r')  # read mode
  h5data = f['proba']
  n_test, n_class, n_samples = h5data.shape
  num_items_per_chunk = h5data.shape[0] // nb_chunks 
  # number of chunks is equal to nb_chunks or nb_chunks + 1
  if h5data.shape[0] % nb_chunks == 0:
    num_chunks = nb_chunks # number of chunks is equal to nb_chunks
  else:
    num_chunks = nb_chunks + 1 # number of chunks is equal to nb_chunks + 1
  print('----------------')
  print('Reading file {} divided in {} chunks.'.format(h5file, num_chunks))
  print('----------------')
  p_mean = []
  for i in np.arange(nb_chunks + 1):
    if i < nb_chunks:
      p_i = h5data[i*num_items_per_chunk:(i+1)*num_items_per_chunk, :, :]
    elif h5data.shape[0] % nb_chunks == 0:  # i == nb_chunks
      # number of chunks is equal to nb_chunks
      break
    else:
      # number of chunks is equal to nb_chunks + 1
      p_i = h5data[i*num_items_per_chunk:, :, :]
    p_mean.append(np.mean(p_i, axis=2))
    
    # Clear memory
    del p_i
    
  p_mean = np.vstack(tuple(p_mean))

  # Clear memory
  gc.collect()
  
  cal = calibration(y, p_mean)
  
  ece = cal['ece']
  
  # Clear memory
  del cal, p_mean
  
  f.close()
  
  # Clear memory
  del h5data
  gc.collect()
  
  return ece


def metrics_ood_from_h5files(h5file_in, h5file_out, nb_chunks=1):
  """Compute metrics out-of-distribution from a (potentially very big) h5file.
  
  Args:
    h5file_in: path to the h5 file containing the tab of probabilities for 
               for in-distribution samples
    h5file_out: path to the h5 file containing the tab of probabilities for
                for out-of-distribution samples
    nb_chunks: int, number of chuncks to divide the h5files
  Returns:
    result_dic: dict that contains the metrics.
  """
  f_in = h5py.File(os.path.join(h5file_in), 'r')  # read mode
  f_out = h5py.File(os.path.join(h5file_out), 'r')  # read mode
  h5data_in = f_in['proba']
  h5data_out = f_out['proba']
  
  # In distribution
  h5data = h5data_in
  n_test, n_class, n_samples = h5data.shape
  num_items_per_chunk = h5data.shape[0] // nb_chunks 
  # number of chunks is equal to nb_chunks or nb_chunks + 1
  if h5data.shape[0] % nb_chunks == 0:
    num_chunks = nb_chunks # number of chunks is equal to nb_chunks
  else:
    num_chunks = nb_chunks + 1 # number of chunks is equal to nb_chunks + 1
  print('----------------')
  print('Reading file {} divided in {} chunks.'.format(h5file_in, num_chunks))
  print('----------------')
  p_mean, p_std, q_tab, p_max_min = [], [], [], []
  for i in np.arange(nb_chunks + 1):
    if i < nb_chunks:
      p_i = h5data[i*num_items_per_chunk:(i+1)*num_items_per_chunk, :, :]
    elif h5data.shape[0] % nb_chunks == 0:  # i == nb_chunks
      # number of chunks is equal to nb_chunks
      break
    else:
      # number of chunks is equal to nb_chunks + 1
      p_i = h5data[i*num_items_per_chunk:, :, :]
    p_mean.append(np.mean(p_i, axis=2))
    p_max_min.append(np.max(p_i, axis=2) - np.min(p_i, axis=2))
    p_std.append(np.std(p_i, axis=2))
    q_tab.append(q_probability(p_i))
    
    # Clear memory
    del p_i
    
  p_mean_in = np.vstack(tuple(p_mean))
  p_max_min_in = np.vstack(tuple(p_max_min))
  p_std_in = np.vstack(tuple(p_std))
  q_tab_in = np.vstack(tuple(q_tab))
  
  del p_mean, p_max_min, p_std, q_tab
  
  # Out of distribution
  h5data = h5data_out
  
  # Clear memory
  gc.collect()
  
  n_test, n_class, n_samples = h5data.shape
  num_items_per_chunk = h5data.shape[0] // nb_chunks 
  # number of chunks is equal to nb_chunks or nb_chunks + 1
  if h5data.shape[0] % nb_chunks == 0:
    num_chunks = nb_chunks # number of chunks is equal to nb_chunks
  else:
    num_chunks = nb_chunks + 1 # number of chunks is equal to nb_chunks + 1
  print('----------------')
  print('Reading file {} divided in {} chunks.'.format(h5file_out, num_chunks))
  print('----------------')
  p_mean, p_std, q_tab, p_max_min = [], [], [], []
  for i in np.arange(nb_chunks + 1):
    if i < nb_chunks:
      p_i = h5data[i*num_items_per_chunk:(i+1)*num_items_per_chunk, :, :]
    elif h5data.shape[0] % nb_chunks == 0:  # i == nb_chunks
      # number of chunks is equal to nb_chunks
      break
    else:
      # number of chunks is equal to nb_chunks + 1
      p_i = h5data[i*num_items_per_chunk:, :, :]
    p_mean.append(np.mean(p_i, axis=2))
    p_max_min.append(np.max(p_i, axis=2) - np.min(p_i, axis=2))
    p_std.append(np.std(p_i, axis=2))
    q_tab.append(q_probability(p_i))
    
  p_mean_out = np.vstack(tuple(p_mean))
  p_max_min_out = np.vstack(tuple(p_max_min))
  p_std_out = np.vstack(tuple(p_std))
  q_tab_out = np.vstack(tuple(q_tab))

  del p_mean, p_max_min, p_std, q_tab
  
  
  result_dic = auroc_aupr(p_mean_in, p_mean_out, 
                          p_max_min_in, p_max_min_out,
                          p_std_in, p_std_out, 
                          q_tab_in, q_tab_out)
  
  f_in.close()
  f_out.close()
  
  del h5data
  del p_max_min_in, p_max_min_out, p_mean_in, p_mean_out
  del p_std_in, p_std_out, q_tab_in, q_tab_out
  gc.collect()
  
  return result_dic


def compute_metrics(y, p_tab):
  """Computation of the metrics.
  
  Args:
    y: numpy array (n_test, num_classes), true observed y, one-hot encoding.
    p_tab: numpy array (n_test, num_classes, num_samples),
           array of probabilities for the num_samples probabilistic 
           classifiers
  """
  mi = entropy(np.mean(p_tab, axis=2)) - np.mean(entropy(p_tab), axis=1)
  p_mean = np.mean(p_tab, axis=2)
  p_std = np.std(p_tab, axis=2)
  p_max_min = np.max(p_tab, axis=2) - np.min(p_tab, axis=2)
  q_tab = q_probability(p_tab)
  ent_q = entropy(q_tab)
  neglog = negloglikelihood(y, p_tab)
  acc = accuracy(y, p_tab)
  bs = brier_score(y, p_tab)
  ent = entropy(p_mean)
  cal = calibration(y, p_mean)
  res_aurc = aurc(y, p_mean, p_std, q_tab, p_max_min)
  risk_cov_std = res_aurc['risk_cov_std']
  risk_cov_softmax = res_aurc['risk_cov_softmax']
  risk_cov_q = res_aurc['risk_cov_q']
  risk_cov_maxmin = res_aurc['risk_cov_maxmin']
  
  result_dic = {'acc': acc,  # n_samples
                'bs': bs,  # n_samples
                'p_mean': p_mean,  # (n_test, n_class)
                'p_std': p_std,  # (n_test, n_class)
                'p_max_min': p_max_min,  # (n_test, n_class)
                'neglog': neglog,  # n_samples
                'ent': ent,  # (n_test, n_samples)
                'cal': cal,  # reliability_diag, ece, mce
                'q_tab': q_tab,  # (n_test, n_class) 
                'ent_q': ent_q,  # n_test
                'mi': mi,  # n_test
                'risk_cov_std': risk_cov_std, # conf, risk_cov, aurc, eaurc
                'risk_cov_softmax': risk_cov_softmax,
                'risk_cov_maxmin': risk_cov_maxmin,
                'risk_cov_q': risk_cov_q
                }
  
  return result_dic


def save_results(result_dic, path_dir, dic_name=None):
  """Save the results using pickle.
  
  Args:
    result_dic: dictionary of the results, output of compute_metrics.
    path_dir: path to the directory where the results are saved.
    dic_name: name of the pickle file
  """
  if dic_name is None:
    dic = 'metrics_dic.pkl'
  else:
    dic = dic_name
  with open(os.path.join(path_dir, dic), 'wb') as handle:
    pickle.dump(result_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
      

def aurc(y, p_mean, p_std, q_tab, p_max_min):
  """Compute the AURC, and other related metrics.
  
  Pairs of (classifier, confidence):
    - (argmax p_mean, - p_std(argmax p_mean))
    - (argmax p_mean, - (p_max_min)(argmax p_mean))
    - (argmax p_mean, max p_mean)
    - (argmax q, -entropy(q))

  Args:
    y: numpy array (n_test, num_classes), one-hot encoding of true observations
    p_mean: numpy array (n_test, num_classes)
            mean of the output probabilities on the test dataset
    p_std: numpy array (n_test, num_classes)
           standard deviation of the output probabilities on the test dataset
    q_tab: numpy array (n_test, num_classes)
           q probability
    p_max_min: numpy array (n_test, num_classes)
               (max - min) of the output probabilities on the test dataset
  Returns:
    dic: dictionary 
         'risk_cov_std': result of sec_classification using 
                         -std of p as a confidence function,
         'risk_cov_softmax': using max p as a confidence function
         'risk_cov_q': using entropy of q as a confidence function.
         'risk_cov_maxmin': using (max - min) of p as a confidence function.
  """
  # Classifier = max p probability
  # Confidence = - std of the max probability along the samples
  y_pred = np.argmax(p_mean, axis=1)
  argmax_y = np.argmax(y, axis=1)
  conf = - p_std[np.arange(p_std.shape[0]), y_pred]
  risk_cov_std = sec_classification(argmax_y, y_pred, conf)
  # Confidence = -(max - min) of the max probability along the samples
  conf = - p_max_min[np.arange(p_max_min.shape[0]), y_pred]
  risk_cov_maxmin = sec_classification(argmax_y, y_pred, conf)
  # Confidence = softmax response
  conf = np.max(p_mean, axis=1)
  risk_cov_softmax = sec_classification(argmax_y, y_pred, conf)
  # Classifier = max q probability
  # Confidence = - entropy of q
  y_pred = np.argmax(q_tab, axis=1)
  conf = - entropy(q_tab)
  risk_cov_q = sec_classification(argmax_y, y_pred, conf)
  
  dic = {'risk_cov_std': risk_cov_std,
         'risk_cov_maxmin' : risk_cov_maxmin,
         'risk_cov_softmax': risk_cov_softmax,
         'risk_cov_q': risk_cov_q}
  
  return dic


def sec_classification(y_true, y_pred, conf):
  """Compute the AURC.

  Args:
    y_true: true labels, vector of size n_test
    y_pred: predicted labels by the classifier, vector of size n_test
    conf: confidence associated to y_pred, vector of size n_test
  Returns:
    dic: dictionary 
      conf: confidence sorted (in decreasing order)
      risk_cov: risk vs coverage (increasing coverage from 0 to 1)
      aurc: AURC
      eaurc: Excess AURC
  """
  n = len(y_true)
  ind = np.argsort(conf)
  y_true, y_pred, conf = y_true[ind][::-1], y_pred[ind][::-1], conf[ind][::-1]
  risk_cov = np.divide(np.cumsum(y_true != y_pred).astype(np.float),
                       np.arange(1, n+1))
  nrisk = np.sum(y_true != y_pred)
  aurc = np.mean(risk_cov)
  opt_aurc = (1./n) * np.sum(np.divide(np.arange(1, nrisk + 1).astype(np.float),
                                       n - nrisk + np.arange(1, nrisk + 1)))
  eaurc = aurc - opt_aurc
  dic = {'conf': conf,
         'risk_cov': risk_cov,
         'aurc': aurc,
         'eaurc': eaurc
        }
  return dic


def sec_classification_comb(y_true, y_pred, conf_1, conf_2):
  """Compute the AURC using a combination of 2 uncertainties measures.

  Args:
    y_true: true labels, vector of size n_test
    y_pred: predicted labels by the classifier, vector of size n_test
    conf_1: one confidence measure associated to y_pred, vector of size n_test
    conf_2: another confidence measure associated 
            to y_pred, vector of size n_test
  Returns:
    results: numpy array of size (3, n_test)
             results[0, :] contains the risk (increasing coverage)
             results[1, :] contains the associated threshold for the 1st 
                           confidence measure
             results[2, :] contains the associated threshold for the 2nd 
                           confidence measure
  """
  n = len(y_true)
  ind_1 = np.argsort(conf_1)[::-1]
  ind_2 = np.argsort(conf_2)[::-1]
  
  conf_1_sorted = conf_1[ind_1]
  conf_2_sorted = conf_2[ind_2]
  
  argsort_ind_1 = np.argsort(ind_1)
  
  indices = argsort_ind_1[ind_2]
  
  
  indices_triang = np.zeros((n, n), dtype=np.uint16)
  risk_triang = np.zeros((n, n), dtype=np.uint16)
  risk_triang[:] = 60000
  
  for i in np.arange(n):
    if i % 1000 == 0:
      print(i)
    if i == 0:
      sort_indices = np.array([indices[0]])
    else:
      j = np.searchsorted(sort_indices, indices[i])
      new_sort_indices = np.zeros(i + 1)
      new_sort_indices[:j] = sort_indices[:j]
      new_sort_indices[j] = indices[i]
      new_sort_indices[(j+1):] = sort_indices[j:]
      sort_indices = new_sort_indices
    indices_triang[i, :(i+1)] = sort_indices.astype(np.uint16)
    ind_risk = ind_1[sort_indices.astype(int)]
    risk_vec = y_true[ind_risk] != y_pred[ind_risk]
    risk_triang[i, :(i+1)] = np.cumsum(risk_vec).astype(np.uint16)
  
  argmin_risk = np.argmin(risk_triang, axis=0)
  min_risk = np.min(risk_triang, axis=0)
  
  results = np.zeros((3, n))
  results[0, :] = np.divide(min_risk.astype(float), 
         np.arange(1, n+1).astype(float))
  results[2, :] = conf_2_sorted[argmin_risk.astype(int)]
  results[1, :] = conf_1_sorted[indices_triang[[argmin_risk.astype(int), 
         np.arange(n)]].astype(int)]
  
  return results

def auroc_aupr(p_mean_in, p_mean_out,
               p_max_min_in, p_max_min_out,
               p_std_in, p_std_out, 
               q_tab_in, q_tab_out):
  # Initialization
  p_mean = np.vstack((p_mean_in, p_mean_out))
  p_max_min = np.vstack((p_max_min_in, p_max_min_out))
  p_std = np.vstack((p_std_in, p_std_out))
  q_tab = np.vstack((q_tab_in, q_tab_out))
  
  # In-distribution, y = 1, out, y = 0
  y = np.concatenate((np.ones(p_mean_in.shape[0], dtype=bool), 
                      np.zeros(p_mean_out.shape[0], dtype=bool)))
  
  # Random permutation
  n = len(y)
  perm = np.random.permutation(n)
  y = y[perm]
  p_mean, p_max_min, p_std, q_tab = p_mean[perm, :], p_max_min[perm, :], \
                                    p_std[perm, :], q_tab[perm, :]
  
  # Conf = softmax
  conf = np.max(p_mean, axis=1)
  auroc_aupr_softmax = out_of_distribution(y, conf)
  # Conf = - std  
  conf = - p_std[np.arange(p_std.shape[0]), np.argmax(p_mean, axis=1)]
  auroc_aupr_std = out_of_distribution(y, conf)
  # Conf = - (max - min)  
  conf = - p_max_min[np.arange(p_max_min.shape[0]), np.argmax(p_mean, axis=1)]
  auroc_aupr_maxmin = out_of_distribution(y, conf)
  # Conf = - entropy of q
  conf = - entropy(q_tab)
  auroc_aupr_q = out_of_distribution(y, conf)
  
  dic = {'auroc_aupr_std': auroc_aupr_std,
         'auroc_aupr_maxmin': auroc_aupr_maxmin,
         'auroc_aupr_softmax': auroc_aupr_softmax,
         'auroc_aupr_q': auroc_aupr_q
        }
  
  return dic
  

def out_of_distribution(y, conf):
  n = len(y)
  ind = np.argsort(conf)
  conf, y = conf[ind], y[ind]
  # Creation of the objects
  # Positive class = in-distribution samples
  tpr = np.zeros(n)
  fpr = np.zeros(n)
  prec_in = np.zeros(n)
  recall_in = np.zeros(n)
  # Step 0
  tp = np.sum(y).astype(float)
  fp = n - tp
  tn = 0.
  fn = 0.
  tpr[0] = tp / (tp + fn)
  fpr[0] = fp / (fp + tn)
  prec_in[0] = tp / (tp + fp)
  recall_in[0] = tp / (tp + fn)
  # Steps 
  for i in np.arange(n - 1):
    if y[i] == 0:
      tn += 1
      fp -= 1
    else:
      fn += 1
      tp -= 1
    tpr[i + 1] = tp / (tp + fn)
    fpr[i + 1] = fp / (fp + tn)
    prec_in[i + 1] = tp / (tp + fp)
    recall_in[i + 1] = tp / (tp + fn)
  # AUROC and AUPR
  ind = np.argsort(recall_in)
  recall_in, prec_in = recall_in[ind], prec_in[ind]
  ind = np.argsort(fpr)
  fpr, tpr = fpr[ind], tpr[ind]
  
  auroc = np.dot(fpr[1:] - fpr[:-1], (tpr[1:] + tpr[:-1]) / 2.)
  aupr_in = np.dot(recall_in[1:] - recall_in[:-1], 
                        (prec_in[1:] + prec_in[:-1]) / 2.)
  
  baseline_aupr_in = np.sum(y).astype(float) / n
  
  # Positive class = out of distribution samples, for AUPR OUT
  prec_out = np.zeros(n)
  recall_out = np.zeros(n)
  # Step 0
  tp = 0.
  fp = 0.
  tn = np.sum(y).astype(float)
  fn = n - tn
  # Steps 
  for i in np.arange(n):
    if y[i] == 0:
      tp += 1
      fn -= 1
    else:
      fp += 1
      tn -= 1
    prec_out[i] = tp / (tp + fp)
    recall_out[i] = tp / (tp + fn)
  # AUPR
  ind = np.argsort(recall_out)
  recall_out, prec_out = recall_out[ind], prec_out[ind]
  aupr_out = np.dot(recall_out[1:] - recall_out[:-1], 
                         (prec_out[1:] + prec_out[:-1]) / 2.)
  
  
  dic = {'tpr': tpr,
         'fpr': fpr,
         'prec_in': prec_in,
         'recall_in': recall_in,
         'prec_out': prec_out,
         'recall_out': recall_out,
         'auroc': auroc,
         'aupr_in': aupr_in,
         'baseline_aupr_in': baseline_aupr_in,
         'aupr_out': aupr_out,
         'baseline_aupr_out': 1 - baseline_aupr_in
        }
  
  return dic


def q_probability(p_tab):
  """Compute the q probability.

  Args:
    p_tab: numpy array, size (?, num_classes, num_samples)
           containing the output predicted probabilities
  Returns:
    q_tab: the probability obtained by averaging the prediction of the ensemble
           of classifiers
  """
  q_tab = np.zeros_like(p_tab)
  d1, _, d2 = p_tab.shape
  q_tab[np.arange(d1).repeat(d2),
        np.argmax(p_tab, axis=1).flatten(), np.tile(np.arange(d2), d1)] = 1.
  q_tab = np.mean(q_tab, axis=2)
  return q_tab


def negloglikelihood(y, p_tab):
  """Compute the negative log-likelihood.

  Args:
    y: one-hot encoding of the true classes, size (?, num_classes)
    p_tab: numpy array, size (?, num_classes, num_samples)
           containing the output predicted probabilities
  Returns:
    neglog: negative log likelihood, along the iterations
            numpy vector of size num_samples
  """
  p_mean = util.cummean(p_tab[y.astype(np.bool), :], axis=1)
  neglog = - np.mean(np.log(p_mean), axis=0)
  return neglog


def accuracy(y, p_tab):
  """Compute the accuracy.

  Args:
    y: one-hot encoding of the true classes, size (?, num_classes)
    p_tab: numpy array, size (?, num_classes, num_samples)
           containing the output predicted probabilities
  Returns:
    acc: accuracy along the iterations, numpy vector of size num_samples
  """
  class_pred = np.argmax(util.cummean(p_tab, axis=2), axis=1)
  argmax_y = np.argmax(y, axis=1)
  acc = np.apply_along_axis(lambda x: np.mean(x == argmax_y),
                            axis=0, arr=class_pred)
  return acc


def brier_score(y, p_tab):
  """Compute the Brier score.

  Brier Score: see
  https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf,
  page 363, Example 1

  Args:
    y: one-hot encoding of the true classes, size (?, num_classes)
    p_tab: numpy array, size (?, num_classes, num_samples)
           containing the output predicted probabilities
  Returns:
    bs: Brier score along the iteration, vector of size num_samples.
  """
  p_cummean = util.cummean(p_tab, axis=2)
  y_repeated = np.repeat(y[:, :, np.newaxis], p_tab.shape[2], axis=2)
  bs = np.mean(np.power(p_cummean - y_repeated, 2), axis=(0, 1))
  return bs


def entropy(p_mean):
  """Compute the entropy.

  Args:
    p_mean: numpy array, size (?, num_classes, ?)
           containing the (possibly mean) output predicted probabilities
  Returns:
    ent: entropy along the iterations, numpy vector of size (?, ?)
  """
  ent = np.apply_along_axis(spstats.entropy, axis=1, arr=p_mean)
  return ent


def calibration(y, p_mean, num_bins=10):
  """Compute the calibration.

  References:
  https://arxiv.org/abs/1706.04599
  https://arxiv.org/abs/1807.00263

  Args:
    y: one-hot encoding of the true classes, size (?, num_classes)
    p_mean: numpy array, size (?, num_classes)
           containing the mean output predicted probabilities
    num_bins: number of bins

  Returns:
    cal: a dictionary
      {reliability_diag: realibility diagram
       ece: Expected Calibration Error
       mce: Maximum Calibration Error
      }
  """
  # Compute for every test sample x, the predicted class.
  class_pred = np.argmax(p_mean, axis=1)
  # and the confidence (probability) associated with it.
  conf = np.max(p_mean, axis=1)
  # Convert y from one-hot encoding to the number of the class
  y = np.argmax(y, axis=1)
  # Storage
  acc_tab = np.zeros(num_bins)  # empirical (true) confidence
  mean_conf = np.zeros(num_bins)  # predicted confidence
  nb_items_bin = np.zeros(num_bins)  # number of items in the bins
  tau_tab = np.linspace(0, 1, num_bins+1)  # confidence bins
  for i in np.arange(num_bins):  # iterate over the bins
    # select the items where the predicted max probability falls in the bin
    # [tau_tab[i], tau_tab[i + 1)]
    sec = (tau_tab[i + 1] > conf) & (conf >= tau_tab[i])
    nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
    # select the predicted classes, and the true classes
    class_pred_sec, y_sec = class_pred[sec], y[sec]
    # average of the predicted max probabilities
    mean_conf[i] = np.mean(conf[sec]) if nb_items_bin[i] > 0 else np.nan
    # compute the empirical confidence
    acc_tab[i] = np.mean(
        class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan

  # Cleaning
  mean_conf = mean_conf[nb_items_bin > 0]
  acc_tab = acc_tab[nb_items_bin > 0]
  nb_items_bin = nb_items_bin[nb_items_bin > 0]

  # Reliability diagram
  reliability_diag = (mean_conf, acc_tab)
  # Expected Calibration Error
  ece = np.average(
      np.absolute(mean_conf - acc_tab),
      weights=nb_items_bin.astype(np.float) / np.sum(nb_items_bin))
  # Maximum Calibration Error
  mce = np.max(np.absolute(mean_conf - acc_tab))
  # Saving
  cal = {'reliability_diag': reliability_diag,
         'ece': ece,
         'mce': mce}
  return cal

#%% Launch functions

def main(argv):
  del argv
  dataset = 'imagenet-first-1000'

  if dataset.startswith('imagenet'):
    y = np.load('saved_models/{}/y.npy'.format(dataset))
  else:
    npzfile = np.load('saved_models/{}/y.npz'.format(dataset))
    y = npzfile['y_test_in']
    
#  h5file_sgd = 'outputs/last_layer/imagenet-first-1000_nbll-1_sgdsgld_lr-0.1_bs-512_s-10/p_sgd_in.h5'
#  h5file_sgld = 'outputs/last_layer/imagenet-first-1000_nbll-1_sgdsgld_lr-0.1_bs-512_s-10/p_sgld_in.h5'
#  h5file_d = 'outputs/last_layer/imagenet-first-1000_nbll-3_dropout_ep-2_lr-0.01_bs-512_s-2_pdrop-0.1/p_in.h5'
  
  for i in [1, 2, 3]:
    h5file = 'outputs/last_layer/imagenet-first-1000_nbll-{}_onepoint/p_in.h5'.format(i)
    path = '/'.join(h5file.split('/')[:-1])
    save_results(metrics_from_h5file(y, h5file), path)
  
#  for h5file, dic_name in zip([h5file_sgd, h5file_sgld, h5file_d],
#                              ['metric_sgd.pkl', 'metrics_sgld.pkl', 
#                               'metrics_dropout.pkl']):
#    path = '/'.join(h5file.split('/')[:-1])
#    save_results(metrics_from_h5file(y, h5file), path, dic_name=dic_name)

  print('End')

if __name__ == '__main__':
  app.run(main)  # mnist, cifar10, cifar100

