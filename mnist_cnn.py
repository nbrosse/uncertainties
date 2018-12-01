#%% Imports

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import numpy as np 

"""
Epoch 19/20
60000/60000 [==============================] - 5s 80us/step - loss: 0.0189 - acc: 0.9941 - val_loss: 0.0266 - val_acc: 0.9925
Epoch 20/20
60000/60000 [==============================] - 5s 83us/step - loss: 0.0206 - acc: 0.9935 - val_loss: 0.0281 - val_acc: 0.9925
Test loss: 0.02807278285419561
Test accuracy: 0.9925
"""

#%% Preprocessing and model

batch_size = 128
num_classes = 10
epochs = 20

metrics = ['accuracy']
loss = keras.losses.categorical_crossentropy
optimizer = keras.optimizers.Adadelta()

# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


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
                       input_shape=input_shape))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax'))

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

#%% Build and train the model

model = build_model(input_shape, num_classes)
saved_model_path = train_model(model, loss, optimizer, metrics, epochs, 
                               batch_size, x_train, y_train, x_test, y_test)

#submodel = Model(inputs=model.input, 
#                 outputs=model.get_layer(index=-3).output)