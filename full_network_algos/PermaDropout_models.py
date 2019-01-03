from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from utils.dropout_layer import PermaDropout
from keras import regularizers


def cifar_model(n_class, p_dropout):
  """Build the network of vgg for cifar.

  Massive dropout and weight decay.

  Args:
    n_class: int, number of classes.
  """
  input_shape = [32, 32, 3]
  WEIGHT_DECAY = 0.0005

  model = Sequential()

  model.add(Conv2D(64, (3, 3), padding='same',
                   input_shape=input_shape,
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(PermaDropout(p_dropout))

  model.add(Conv2D(64, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(128, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(PermaDropout(p_dropout))

  model.add(Conv2D(128, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(256, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(PermaDropout(p_dropout))

  model.add(Conv2D(256, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(PermaDropout(p_dropout))

  model.add(Conv2D(256, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(512, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(PermaDropout(p_dropout))

  model.add(Conv2D(512, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(PermaDropout(p_dropout))

  model.add(Conv2D(512, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(512, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(PermaDropout(p_dropout))

  model.add(Conv2D(512, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(PermaDropout(p_dropout))

  model.add(Conv2D(512, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(PermaDropout(p_dropout))

  model.add(Flatten())
  model.add(Dense(512, kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization(name='features_layer'))

  model.add(PermaDropout(p_dropout))
  model.add(Dense(n_class, name='ll_dense'))
  model.add(Activation('softmax'))
  return model

def mnist_model(num_classes,p_dropout):
  model = Sequential()
  model.add(Flatten(input_shape=(28, 28), name='l_1'))
  model.add(Dense(512, activation='relu', name='l_2'))
  model.add(PermaDropout(p_dropout, name='dropout_1'))
  model.add(Dense(20, activation='relu', name='features_layer'))
  model.add(PermaDropout(p_dropout, name='dropout_2'))
  model.add(Dense(num_classes, activation='softmax', name='ll_dense'))
  return model