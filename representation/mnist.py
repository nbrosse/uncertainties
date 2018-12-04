# -*- coding: utf-8 -*-
"""Simple feedforward neural network for Mnist."""


#%% Packages

import os
import shutil
from absl import app

import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense, Flatten
import utils.util as util

"""
losses in
[0.04282997509721845, 0.9916326133489005]
"""

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


def main(argv):
  # batch_size = 32
  n_class = argv[0]
  model = build_model(n_class)
  (x_train, y_train), (x_test, y_test) = input_data()
  sec, index = util.select_classes(y_train, n_class)
  
  path_dir = 'saved_models/mnist_sec_{}'.format(n_class)
  if os.path.isdir(path_dir):
    shutil.rmtree(path_dir, ignore_errors=True)
  os.makedirs(path_dir)
  np.save(os.path.join(path_dir, 'index.npy'), index)

  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(x_train[sec,:], y_train[np.ix_(sec, index)], epochs=20)
  model_path = os.path.join(path_dir, 'mnist.h5')
  model.save(model_path)
  print('Saved trained model at %s ' % model_path)
  
  sec = np.dot(y_test, index).astype(bool)

  losses_in = model.evaluate(x_test[sec,:], y_test[np.ix_(sec, index)])
  
  print('losses in')
  print(losses_in)

if __name__ == '__main__':
  app.run(main, argv=[5])
