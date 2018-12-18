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
import os
import numpy as np
import scipy.stats as spstats
import utils.util as util

#%% functions to compute the metrics


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


def save_results(result_dic, path_dir):
  """Save the results using pickle.
  
  Args:
    result_dic: dictionary of the results, output of compute_metrics.
    path_dir: path to the directory where the results are saved.
  """
  with open(os.path.join(path_dir, 'metrics_dic.pkl'), 'wb') as handle:
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






