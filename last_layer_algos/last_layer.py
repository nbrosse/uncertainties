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

def features_extraction(model, model_path, input_data):
  """Extract the features from the last layer.
  
  Args: 
    model: full keras model.
    model_path: path to the full saved model.
    input_data: input data for the features.
  Returns:
    features: features corresponding to the input data.
  """
  submodel = Model(inputs=model.input, 
                   outputs=model.get_layer('features_layer').output)
  submodel.load_weights(model_path, by_name=True)
  features = submodel.predict(input_data)
  return features


def build_last_layer(model_path, features_shape, num_classes, p_dropout=None):
  """Build the last layer keras model.
  
  Args:
    model_path: path to the full saved model.
    features_shape: shape of the features.
    num_classes: int, number of classes.
    p_dropout: float between 0 and 1. Fraction of the input units to drop.
  Returns:
    submodel: last layer model.
  """
  if p_dropout is not None:
    x = Input(shape=features_shape, name='ll_input')
    y = PermaDropout(p_dropout, name='ll_dropout')(x)
    y = Dense(num_classes, activation='softmax', name='ll_dense')(y)
    model = Model(inputs=x, outputs=y)
  else:
    model = Sequential()
    model.add(Dense(num_classes, activation='softmax', 
                    input_shape=features_shape, name='ll_dense'))
  model.load_weights(model_path, by_name=True)
  return model

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
  # Compile and train model
  model.compile(optimizer=optimizer, 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
  # Saving after every N batches
  # https://stackoverflow.com/questions/43794995/python-keras-saving-model-weights-after-every-n-batches
  mc = keras.callbacks.ModelCheckpoint(os.path.join(path_weights, 
                                                    'weights{epoch:03d}.h5'),
                                       save_weights_only=True, 
                                       period=thinning_interval)

  hist = model.fit(features_train, y_train,
                   batch_size=batch_size,
                   epochs=epochs,
                   verbose=1,
                   validation_data=(features_test, y_test),
                   callbacks=[mc])
  # Sanity check
  score = model.evaluate(features_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
  
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