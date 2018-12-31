# -*- coding: utf-8 -*-
"""Simple feedforward neural network for Mnist."""


#%% Packages

import os
import shutil
from absl import app

import numpy as np
import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
import utils.util as util

NUM_CLASSES = 10

#%% Define and train the model

def build_model(num_classes):
  model = Sequential()
  model.add(Flatten(input_shape=(28, 28), name='l_1'))
  model.add(Dense(512, activation='relu', name='l_2'))
  model.add(Dense(20, activation='relu', name='features_layer'))
  model.add(Dense(num_classes, activation='softmax', name='ll_dense'))
  return model


def input_data():
  (x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  y_train = keras.utils.to_categorical(y_train, 10)
  y_test = keras.utils.to_categorical(y_test, 10)
  return (x_train, y_train), (x_test, y_test)


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


def main(argv):
  # batch_size = 32, default
  # Training of the model
  n_class = argv[0]
  method = argv[1]
  (x_train, y_train), (x_test, y_test) = input_data()
  model = build_model(n_class)
  index = util.select_classes(y_train, n_class, method=method)
  path_dir = 'saved_models/mnist-{}-{}'.format(method, n_class)
  sec = np.dot(y_train, index).astype(bool)
  
  if os.path.isdir(path_dir):
    shutil.rmtree(path_dir, ignore_errors=True)
  os.makedirs(path_dir)
  
  np.save(os.path.join(path_dir, 'index.npy'), index)

  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  
  model.fit(x_train[sec,:], y_train[np.ix_(sec, index)], epochs=20)
  model_path = os.path.join(path_dir, 'mnist.h5')
  model.save_weights(model_path)
  print('Saved trained model at %s ' % model_path)
  
  sec_test = np.dot(y_test, index).astype(bool)

  losses_in = model.evaluate(x_test[sec_test,:], 
                             y_test[np.ix_(sec_test, index)])
  print('losses in')
  print(losses_in)

  # In and out of sample distribution
  sec_train = np.dot(y_train, index).astype(bool)
  sec_test = np.dot(y_test, index).astype(bool)
  
  x_train_in, x_train_out = x_train[sec_train, :], x_train[~sec_train, :]
  y_train_in = y_train[np.ix_(sec_train, index)]
  x_test_in, x_test_out = x_test[sec_test, :], x_test[~sec_test, :]
  y_test_in = y_test[np.ix_(sec_test, index)]
  
  # Compute the features
  submodel = Model(inputs=model.input, 
                   outputs=model.get_layer('features_layer').output)
  submodel.load_weights(model_path, by_name=True)
  
  features_train_in = submodel.predict(x_train_in)
  features_train_out = submodel.predict(x_train_out)
  features_test_in = submodel.predict(x_test_in)
  features_test_out = submodel.predict(x_test_out)
  
  np.savez(os.path.join(path_dir, 'features.npz'), 
           features_train_in=features_train_in,
           features_train_out=features_train_out,
           features_val_in=features_test_in,
           features_val_out=features_test_out
          )
  
  np.savez(os.path.join(path_dir, 'y.npz'),
           y_train_in=y_train_in,
           y_val_in=y_test_in
          )


if __name__ == '__main__':
  app.run(main, argv=[5, 'first'])

