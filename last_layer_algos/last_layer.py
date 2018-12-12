#%% Imports

from __future__ import print_function

import os
import glob
import keras

from keras.models import Model, Sequential
from keras.layers import Dense, Input
from utils.dropout_layer import PermaDropout

import numpy as np


#%% Features, last layer model






def sgd_sgld_last_layer(model, optimizer, epochs, batch_size, 
                        features_train, y_train, features_test, y_test,
                        thinning_interval, path_weights):
  """Train last layer model using SGD and SGLD.
  Weights snapshots every thinning_interval.
  
  Args:
    model: keras model.
    optimizer: optimizer
    epochs: epochs
    batch_size: batch_size
    features_train: features_train of the last layer
    y_train: y_train
    features_test: features_test of the last layer
    y_test: y_test
    thinning_interval: int, thinning interval between snapshots.
    path_weights: str, directory where to write snapshots of the weights
  Returns:
    hist: history (keras) object of the training 
  """

  
  return hist


def bootstrap_last_layer(model, epochs, batch_size, 
                         bootstrap_features_train, bootstrap_y_train, 
                         features_test, y_test,
                         model_path):
  """Train last layer model on a boostrap training sample
  
  Args:
    model: keras model.
    epochs: epochs
    batch_size: batch_size
    bootstrap_features_train: features_train boostrapped of the last layer
    bootstrap_y_train: y_train boostrapped of the last layer
    features_test: features_test of the last layer
    y_test: y_test
    model_path: path to the model where initial weights are saved
  """
  # Load weights
  model.load_weights(model_path, by_name=True)
  # Train
  model.fit(bootstrap_features_train, bootstrap_y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(features_test, y_test))
  # Sanity check
  score = model.evaluate(features_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
  
  return model

def predict_sgd_sgld_last_layer(model, features_test, 
                                num_classes, path_weights):
  """Predict probability distributions using saved weights.
  """
  weights_files = [i for i in glob.glob(os.path.join(path_weights, '*.h5'))]
  num_samples = len(weights_files)
  proba_tab = np.zeros(shape=(features_test.shape[0], 
                              num_classes, 
                              num_samples))
  for index, weights in enumerate(weights_files):
      model.load_weights(weights)
      proba = model.predict(features_test)
      proba_tab[:, :, index] = proba

  return proba_tab