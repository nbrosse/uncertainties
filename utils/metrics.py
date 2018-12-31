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
- AURC and related metrics.
"""

#%% Imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
from absl import app
import os
import h5py
from psutil import virtual_memory

import numpy as np
import scipy.stats as spstats
import utils.util as util

mem = virtual_memory()
MEM = mem.total / 40.0 # (2.0 * nb_cores) # max physical memory for us

#%% functions to compute the metrics

def metrics_from_h5file(y, h5file):
  """Compute metrics from a (potentially very big) h5file.
  
  Args:
    y: true y, one-hot encoding size (n_test, n_class)
    h5file: path to the h5 file containing the tab of probabilities.
  Returns:
    result_dic: dict that contains the metrics.
  """
  memsize = os.path.getsize(h5file)
  nb_chunks = int(memsize // MEM + 1) 
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
  p_mean, p_std, q_tab = [], [], []
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
    p_std.append(res_dic['p_std'])
    q_tab.append(res_dic['q_tab'])
    acc[i, :] = res_dic['acc']
    bs[i, :] = res_dic['bs']
    neglog[i, :] = res_dic['neglog']
    ent.append(res_dic['ent'])
    ent_q.append(res_dic['ent_q'])
    mi.append(res_dic['mi'])
    
  acc = np.mean(acc, axis=0)
  bs = np.mean(bs, axis=0)
  neglog = np.mean(neglog, axis=0)

  p_mean = np.vstack(tuple(p_mean))
  p_std = np.vstack(tuple(p_std))
  q_tab = np.vstack(tuple(q_tab))
  ent = np.concatenate(tuple(ent))
  ent_q = np.concatenate(tuple(ent_q))
  mi = np.concatenate(tuple(mi))
  
  cal = calibration(y, p_mean)
  
  risk_cov = aurc(y, p_mean, p_std, q_tab)
  
  result_dic = {}
  
  result_dic = {'acc': acc,  # n_samples
              'bs': bs,  # n_samples
              'p_mean': p_mean,  # (n_test, n_class)
              'p_std': p_std,  # (n_test, n_class)
              'neglog': neglog,  # n_samples
              'ent': ent,  # (n_test, n_samples)
              'cal': cal,  # reliability_diag, ece, mce
              'q_tab': q_tab,  # (n_test, n_class) 
              'ent_q': ent_q,  # n_test
              'mi': mi,  # n_test
              'risk_cov_std': risk_cov['risk_cov_std'], # conf, risk_cov, aurc, eaurc
              'risk_cov_softmax': risk_cov['risk_cov_softmax'],
              'risk_cov_q': risk_cov['risk_cov_q']
              }
  
  f.close()
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
  q_tab = q_probability(p_tab)
  ent_q = entropy(q_tab)
  neglog = negloglikelihood(y, p_tab)
  acc = accuracy(y, p_tab)
  bs = brier_score(y, p_tab)
  ent = entropy(p_mean)
  cal = calibration(y, p_mean)
  res_aurc = aurc(y, p_mean, p_std, q_tab)
  risk_cov_std = res_aurc['risk_cov_std']
  risk_cov_softmax = res_aurc['risk_cov_softmax']
  risk_cov_q = res_aurc['risk_cov_q']
  
  result_dic = {'acc': acc,  # n_samples
                'bs': bs,  # n_samples
                'p_mean': p_mean,  # (n_test, n_class)
                'p_std': p_std,  # (n_test, n_class)
                'neglog': neglog,  # n_samples
                'ent': ent,  # (n_test, n_samples)
                'cal': cal,  # reliability_diag, ece, mce
                'q_tab': q_tab,  # (n_test, n_class) 
                'ent_q': ent_q,  # n_test
                'mi': mi,  # n_test
                'risk_cov_std': risk_cov_std, # conf, risk_cov, aurc, eaurc
                'risk_cov_softmax': risk_cov_softmax,
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
      

def aurc(y, p_mean, p_std, q_tab):
  """Compute the AURC, and other related metrics.
  
  Pairs of (classifier, confidence):
    - (argmax p_mean, - p_std(argmax p_mean))
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
  Returns:
    dic: dictionary 
         'risk_cov_std': result of sec_classification using 
                         -std of p as a confidence function,
         'risk_cov_softmax': using max p as a confidence function
         'risk_cov_q': using entropy of q as a confidence function.
  """
  # Classifier = max p probability
  # Confidence = - std of the max probability along the samples
  y_pred = np.argmax(p_mean, axis=1)
  argmax_y = np.argmax(y, axis=1)
  conf = - p_std[np.arange(p_std.shape[0]), y_pred]
  risk_cov_std = sec_classification(argmax_y, y_pred, conf)
  # Confidence = softmax response
  conf = np.max(p_mean, axis=1)
  risk_cov_softmax = sec_classification(argmax_y, y_pred, conf)
  # Classifier = max q probability
  # Confidence = - entropy of q
  y_pred = np.argmax(q_tab, axis=1)
  conf = - entropy(q_tab)
  risk_cov_q = sec_classification(argmax_y, y_pred, conf)
  
  dic = {'risk_cov_std': risk_cov_std,
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

#%% launch functions

def main(argv):
  del argv
#  dataset = 'cifar100-first-100'
  dataset = 'imagenet-first-1000'
  # y
#  npzfile = np.load('saved_models/{}/y.npz'.format(dataset))
#  y = npzfile['y_test_in']
  # y imagenet
  y = np.load('saved_models/{}/y.npy'.format(dataset))
  
  # Paths to folders
#  path_sgd_sgld = 'outputs/last_layer/{}_sgdsgld_lr-0.001_bs-128_s-1000'.format(dataset)
#  path_dropout = 'outputs/last_layer/{}_dropout_ep-100_lr-0.005_bs-128_s-100_pdrop-0.5'.format(dataset)
#  path_bootstrap = 'outputs/last_layer/{}_bootstrap_ep-10_lr-0.005_bs-128_s-10'.format(dataset)
#  path_onepoint = 'outputs/last_layer/{}_onepoint'.format(dataset)
#  path_sgd = 'outputs/last_layer/imagenet-first-1000_sgdsgld_lr-0.01_bs-512_s-100'
  path = 'outputs/last_layer/imagenet-first-1000_dropout_ep-10_lr-0.01_bs-512_s-10_pdrop-0.1'
  
  # Paths to h5 files
  p = os.path.join(path, 'p_in.h5')
#  p_sgld = os.path.join(path_sgd_sgld, 'p_sgld_in.h5')
#  p_dropout = os.path.join(path_dropout, 'p_in.h5')
#  p_bootstrap = os.path.join(path_bootstrap, 'p_in.h5')
#  p_onepoint = os.path.join(path_onepoint, 'p_in.h5')
  
  save_results(metrics_from_h5file(y, p), path)
#  save_results(metrics_from_h5file(y, p_sgld), path_sgd_sgld, 'dic_sgld.pkl')
#  save_results(metrics_from_h5file(y, p_dropout), path_dropout)
#  save_results(metrics_from_h5file(y, p_bootstrap), path_bootstrap)
#  save_results(metrics_from_h5file(y, p_onepoint), path_onepoint)
  print('End')


if __name__ == '__main__':
  app.run(main)  # mnist, cifar10, cifar100

