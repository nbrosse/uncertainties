# -*- coding: utf-8 -*-
"""Ensemble feedforward neural network for Mnist."""

# Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles

#%% Packages

import os
import gc
import shutil
from absl import app

import numpy as np
import keras

K = keras.backend

from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization
from utils.dropout_layer import PermaDropout

NUM_CLASSES = 10
NUM_TEST_MNIST = 10000
NUM_TEST_NOTMNIST = 10000

#%% Define and train the model

def build_model(num_classes=10, p_dropout=None):
  if p_dropout is None:
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28), name='input_1'))
    model.add(Dense(200, activation='relu', name='dense_1'))
    model.add(BatchNormalization(name='bnorm_1'))
    model.add(Dense(200, activation='relu', name='dense_2'))
    model.add(BatchNormalization(name='bnorm_2'))
    model.add(Dense(200, activation='relu', name='dense_3'))
    model.add(BatchNormalization(name='bnorm_3'))
    model.add(Dense(num_classes, activation='softmax', name='dense_4'))
  else:
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28), name='input_1'))
    model.add(Dense(200, activation='relu', name='dense_1'))
    model.add(BatchNormalization(name='bnorm_1'))
    model.add(PermaDropout(p_dropout, name='dropout_1'))
    model.add(Dense(200, activation='relu', name='dense_2'))
    model.add(BatchNormalization(name='bnorm_2'))
    model.add(PermaDropout(p_dropout, name='dropout_2'))
    model.add(Dense(200, activation='relu', name='dense_3'))
    model.add(BatchNormalization(name='bnorm_3'))
    model.add(PermaDropout(p_dropout, name='dropout_3'))
    model.add(Dense(num_classes, activation='softmax', name='dense_4'))
  return model


def reset_weights(model):
  session = K.get_session()
  for layer in model.layers: 
    if hasattr(layer, 'kernel_initializer'):
      layer.kernel.initializer.run(session=session)
    if hasattr(layer, 'bias_initializer'):
      layer.bias.initializer.run(session=session)


def input_data():
  (x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  y_train = keras.utils.to_categorical(y_train, 10)
  y_test = keras.utils.to_categorical(y_test, 10)
  return (x_train, y_train), (x_test, y_test)

def input_data_notmnist():
  x_test = np.load('notmnist/x_test_notmnist.npy') # already preprocessed
  y_test = np.load('notmnist/y_test_notmnist.npy')
  y_test = keras.utils.to_categorical(y_test, 10)
  return (x_test, y_test)
  

def training(argv):
  (x_train, y_train), (x_test, y_test) = input_data()
  model = build_model()
  path_dir = 'outputs/ensemble_mnist'
  
  if os.path.isdir(path_dir):
    shutil.rmtree(path_dir, ignore_errors=True)
  os.makedirs(path_dir)
  
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  
  for i in np.arange(10):
    
    print('--------------')
    print('Step {}'.format(i))
    print('--------------')
    
    epochs = 10
    
    model.fit(x_train, y_train, epochs=epochs,
              validation_data=(x_test, y_test))
    model_path = os.path.join(path_dir, 
                              'ensemble_ep-{}_{}.h5'.format(epochs, i))
    model.save_weights(model_path)
    print('Saved trained model at %s ' % model_path)
    
    reset_weights(model)

  del model
  gc.collect()
  
  model_d = build_model(p_dropout=0.1)
  
  model_d.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])  
  
  print('--------------')
  print('Dropout training')
  print('--------------')
  
  model_d.fit(x_train, y_train, epochs=epochs,
              validation_data=(x_test, y_test))
  model_path = os.path.join(path_dir, 'dropout_ep-{}.h5'.format(epochs))
  model_d.save_weights(model_path)
  print('Saved trained model at %s ' % model_path)
  
  del model_d
  gc.collect()
  
  print('End')
  

def testing(hparams):
  (_, _), (x_mnist, y_mnist) = input_data()
  (x_notmnist, y_notmnist) = input_data_notmnist()
  
  x_test = np.vstack((x_mnist, x_notmnist))
  y_test = np.vstack((y_mnist, y_notmnist))
  
  shape_in = (x_test.shape[0], NUM_CLASSES, 10)
  proba = np.zeros(shape_in)
  
  model = build_model()
  path_dir = 'outputs/ensemble_mnist'
  
  # Not needed normally 
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  
  epochs = 10
  
  for i in np.arange(10):
    
    print('--------------')
    print('Step {}'.format(i))
    print('--------------')
    
    model_path = os.path.join(path_dir, 
                              'ensemble_ep-{}_{}.h5'.format(epochs, i))
    model.load_weights(model_path)
    print('Loaded model at %s ' % model_path)
    
    proba[:, :, i] = model.predict(x_test)

  del model
  gc.collect()

if __name__ == '__main__':
  app.run(training)

