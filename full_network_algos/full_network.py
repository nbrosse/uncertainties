#%% Imports

from __future__ import print_function

import os
import glob
import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from utils.dropout_layer import PermaDropout

import numpy as np


#%% Full networks with PermaDropout layers

def build_model_cifar10(x_train, num_classes, p_dropout):
  model = Sequential()
  model.add(Conv2D(32, (3, 3), padding='same',
                   input_shape=x_train.shape[1:],
                   name='l_1'))
  model.add(Activation('relu', name='l_2'))
  model.add(Conv2D(32, (3, 3), name='l_3'))
  model.add(Activation('relu', name='l_4'))
  model.add(MaxPooling2D(pool_size=(2, 2), name='l_5'))
  model.add(PermaDropout(p_dropout, name='l_dropout_1'))
  
  model.add(Conv2D(64, (3, 3), padding='same', name='l_7'))
  model.add(Activation('relu', name='l_8'))
  model.add(Conv2D(64, (3, 3), name='l_9'))
  model.add(Activation('relu', name='l_10'))
  model.add(MaxPooling2D(pool_size=(2, 2), name='l_11'))
  model.add(PermaDropout(p_dropout, name='l_dropout_2'))
  
  model.add(Flatten(name='l_13'))
  model.add(Dense(512, activation='relu', name='features_layer'))
  model.add(PermaDropout(p_dropout, name='l_dropout_3'))
  model.add(Dense(num_classes, activation='softmax', name='ll_dense'))
  return model

def build_model_cifar100(x_train, num_classes, p_dropout):
  model = Sequential()
  
  model.add(Conv2D(128, (3, 3), padding='same',
                   input_shape=x_train.shape[1:],
                   name='l_1'))
  model.add(Activation('elu', name='l_2'))
  model.add(Conv2D(128, (3, 3), name='l_3'))
  model.add(Activation('elu', name='l_4'))
  model.add(MaxPooling2D(pool_size=(2, 2), name='l_5'))
  
  model.add(Conv2D(256, (3, 3), padding='same', name='l_6'))
  model.add(Activation('elu', name='l_7'))
  model.add(Conv2D(256, (3, 3), name='l_8'))
  model.add(Activation('elu', name='l_9'))
  model.add(MaxPooling2D(pool_size=(2, 2), name='l_10'))
  model.add(PermaDropout(p_dropout, name='l_dropout_1'))
  
  model.add(Conv2D(512, (3, 3), padding='same', name='l_12'))
  model.add(Activation('elu', name='l_13'))
  model.add(Conv2D(512, (3, 3), name='l_14'))
  model.add(Activation('elu', name='l_15'))
  model.add(MaxPooling2D(pool_size=(2, 2), name='l_16'))
  model.add(PermaDropout(p_dropout, name='l_dropout_2'))
  
  
  model.add(Flatten(name='l_18'))
  model.add(Dense(1024, activation='elu', name='features_layer'))
  model.add(PermaDropout(p_dropout, name='l_dropout_3'))
  model.add(Dense(num_classes, activation='softmax', name='ll_dense'))
  return model

def build_model_mnist(p_dropout):
  model = Sequential()
  model.add(Flatten(input_shape=(28, 28), name='l_1'))
  model.add(Dense(512, activation='relu', name='l_2'))
  model.add(PermaDropout(p_dropout, name='l_dropout_1'))
  model.add(Dense(20, activation='relu', name='features_layer'))
  model.add(PermaDropout(p_dropout, name='l_dropout_2'))
  model.add(Dense(10, activation='softmax', name='ll_dense'))
  return model 

#%% sgd_sgld on the full network

def sgd_sgld(model, optimizer, epochs, batch_size, 
             x_train, y_train, x_test, y_test,
             thinning_interval, path_weights):
  """Train full model using SGD or SGLD.
  Weights snapshots every thinning_interval.
  
  Args:
    model: keras model.
    optimizer: optimizer
    epochs: epochs
    batch_size: batch_size
    x_train: features_train of the last layer
    y_train: y_train
    x_test: features_test of the last layer
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

  hist = model.fit(x_train, y_train,
                   batch_size=batch_size,
                   epochs=epochs,
                   verbose=1,
                   validation_data=(x_test, y_test),
                   callbacks=[mc])
  # Sanity check
  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
  
  return hist

def predict_sgd_sgld(model, x_test, 
                     num_classes, path_weights):
  """Predict probability distributions using saved weights.
  """
  weights_files = [i for i in glob.glob(os.path.join(path_weights, '*.h5'))]
  num_samples = len(weights_files)
  proba_tab = np.zeros(shape=(x_test.shape[0], 
                              num_classes, 
                              num_samples))
  for index, weights in enumerate(weights_files):
      model.load_weights(weights)
      proba = model.predict(x_test)
      proba_tab[:, :, index] = proba

  return proba_tab