#%% Imports

from __future__ import print_function
from absl import app

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import numpy as np 

"""
Epoch 19/20
60000/60000 [==============================] - 5s 83us/step - loss: 0.0022 - acc: 0.9994 - val_loss: 0.0427 - val_acc: 0.9910
Epoch 20/20
60000/60000 [==============================] - 5s 79us/step - loss: 0.0021 - acc: 0.9995 - val_loss: 0.0377 - val_acc: 0.9925
Test loss: 0.03771045614296222
Test accuracy: 0.9925
"""

#%% Preprocessing and model

def input_data():
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  x_train = x_train[:, :, :, np.newaxis]
  x_test = x_test[:, :, :, np.newaxis]
  # convert class vectors to binary class matrices
  y_train = keras.utils.to_categorical(y_train, 10)
  y_test = keras.utils.to_categorical(y_test, 10)
  return (x_train, y_train), (x_test, y_test)


def build_model(input_shape, num_classes):
  """pre-training phase on CNN model.
  
  (From F.Chollet book > 95% accuracy with 12 epochs)
  Args:
    input_shape: shape of the input, channels last, (img_rows, img_cols, 1)
    num_classes: int, number of classes.
  Returns:
    model: keras cnn model
  """
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3),
                       activation='relu',
                       input_shape=input_shape,
                       name='l_1'))
  model.add(Conv2D(64, (3, 3), activation='relu', name='l_2'))
  model.add(MaxPooling2D(pool_size=(2, 2), name='l_3'))
  model.add(Dropout(0.25, name='l_4'))
  model.add(Flatten(name='l_5'))
  model.add(Dense(128, activation='relu', name='features_layer'))
#  model.add(Dropout(0.5)) # Remove because of last layer
  model.add(Dense(num_classes, activation='softmax', name='ll_dense'))

  return model

def train_model(model, loss, optimizer, metrics, epochs, batch_size, 
                x_train, y_train, x_test, y_test):
  """Train the model.
  
  Args:
    model: keras model.
    loss: loss
    optimizer: optimizer
    metrics: metrics
    epochs: epochs
    batch_size: batch_size
    x_train: x_train
    y_train: y_train
    x_test: x_test
    y_test: y_test
  Returns:
    model_path: path to the saved model.
  """
  model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
  model_path='saved_models/mnist_cnn.h5'
  model.save(model_path)
  return model_path

#%% Train the model

def main(_):
  batch_size = 128
  epochs = 20
  num_classes = 10
  
  metrics = ['accuracy']
  loss = keras.losses.categorical_crossentropy
  optimizer = keras.optimizers.Adadelta()
  
  # input image dimensions
  img_rows, img_cols = 28, 28
  input_shape = (img_rows, img_cols, 1)

  (x_train, y_train), (x_test, y_test) = input_data()
  
  model = build_model(input_shape, num_classes)
  model_path = train_model(model, loss, optimizer, metrics, epochs, 
                           batch_size, x_train, y_train, x_test, y_test)

if __name__ == '__main__':
  app.run(main)