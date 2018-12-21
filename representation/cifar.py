
#%% Packages

from __future__ import print_function

from absl import app
import os
import shutil

import numpy as np

import keras
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras import regularizers

from sklearn.model_selection import train_test_split

import utils.util as util

WEIGHT_DECAY = 0.0005
SHAPE = [32, 32, 3]

#%% Model

def input_data(dataset):
  if dataset == 'cifar10':
    (x_train, y_train),(x_test, y_test) = cifar10.load_data()
    num_classes = 10
  else:
    (x_train, y_train),(x_test, y_test) = cifar100.load_data()
    num_classes = 100
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)
  return (x_train, y_train), (x_test, y_test)

def build_model(n_class):
  """Build the network of vgg for cifar.
  
  Massive dropout and weight decay.
  
  Args:
    n_class: int, number of classes.
  """

  model = Sequential()

  model.add(Conv2D(64, (3, 3), padding='same', 
                   input_shape=SHAPE,
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.3))

  model.add(Conv2D(64, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(128, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))

  model.add(Conv2D(128, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(256, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))

  model.add(Conv2D(256, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))

  model.add(Conv2D(256, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(MaxPooling2D(pool_size=(2, 2)))


  model.add(Conv2D(512, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))

  model.add(Conv2D(512, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))

  model.add(Conv2D(512, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(512, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))

  model.add(Conv2D(512, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))

  model.add(Conv2D(512, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.5))

  model.add(Flatten())
  model.add(Dense(512,kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization(name='features_layer'))

  model.add(Dropout(0.5))
  model.add(Dense(n_class, name='ll_dense'))
  model.add(Activation('softmax'))
  return model


def whitening(x_train):
  """Normalize inputs for zero mean and unit variance.
  
  Args:
    x_train: training dataset,
  Returns:
    x_train: normalized training dataset.
    mean, std = mean and std 
  """
  mean = np.mean(x_train, axis=(0, 1, 2, 3))
  std = np.std(x_train, axis=(0, 1, 2, 3))
  x_train = (x_train - mean) / (std + 1e-7)
  return x_train, (mean, std)

def normalize(x, mean, std):
  """Normalize inputs on the test or validation dataset.
  
  Args:
    x: inputs
    mean, std = mean and std
  Returns:
    x: normalized inputs.
  """
  return (x - mean)/(std + 1e-7)


def train(model, x_train, y_train, x_test, y_test):
  """ Training the model"""
  
  # training parameters
  batch_size = 128
  maxepoches = 250
  learning_rate = 0.1
  lr_decay = 1e-6
  lr_drop = 20


  def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))
  reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

  #data augmentation
  datagen = ImageDataGenerator(
      featurewise_center=False,  # set input mean to 0 over the dataset
      samplewise_center=False,  # set each sample mean to 0
      featurewise_std_normalization=False,  # divide inputs by std of the dataset
      samplewise_std_normalization=False,  # divide each input by its std
      zca_whitening=False,  # apply ZCA whitening
      rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
      width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
      height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
      horizontal_flip=True,  # randomly flip images
      vertical_flip=False)  # randomly flip images
  # (std, mean, and principal components if ZCA whitening is applied).
  datagen.fit(x_train)



  #optimization details
  sgd = optimizers.SGD(lr=learning_rate, 
                       decay=lr_decay, 
                       momentum=0.9, 
                       nesterov=True)
  model.compile(loss='categorical_crossentropy', 
                optimizer=sgd,
                metrics=['accuracy'])


  # training process in a for loop with learning rate drop every 25 epoches
  historytemp = model.fit_generator(datagen.flow(x_train, y_train,
                                                 batch_size=batch_size), 
                      steps_per_epoch=x_train.shape[0] // batch_size,
                      epochs=maxepoches,
                      validation_data=(x_test, y_test),
                      callbacks=[reduce_lr],
                      verbose=2)
  return model, historytemp

#%% main function

def main(argv):
  n_class = argv[0]
  method = argv[1]
  dataset = argv[2]
  (x_train, y_train), (x_test, y_test) = input_data(dataset)
  x_train, (mean, std) = whitening(x_train)
  x_test = normalize(x_test, mean, std)
  model = build_model(n_class)
  index = util.select_classes(y_train, n_class, method=method)
  path_dir = 'saved_models/{}-{}-{}'.format(dataset, method, n_class)
  
  sec = np.dot(y_train, index).astype(bool)
  sec_test = np.dot(y_test, index).astype(bool)
  
  if os.path.isdir(path_dir):
    os.chmod(path_dir, 0o777)
    shutil.rmtree(path_dir, ignore_errors=True)
  os.makedirs(path_dir)
  
  np.save(os.path.join(path_dir, 'index.npy'), index)
  
  model, hist = train(model, 
                      x_train[sec,:], y_train[np.ix_(sec, index)], 
                      x_test[sec_test,:], y_test[np.ix_(sec_test, index)])
  
  model_path = os.path.join(path_dir, '{}.h5'.format(dataset))
  model.save_weights(model_path)
  print('Saved trained model at %s ' % model_path)
  
  

  losses_in = model.evaluate(x_test[sec_test,:], 
                             y_test[np.ix_(sec_test, index)])
  print('losses in')
  print(losses_in)

  # Splitting train into train and validation sets.
  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train)
  
  # In and out of sample distribution
  sec_train = np.dot(y_train, index).astype(bool)
  sec_val = np.dot(y_val, index).astype(bool)
  sec_test = np.dot(y_test, index).astype(bool)
  
  x_train_in, x_train_out = x_train[sec_train, :], x_train[~sec_train, :]
  y_train_in = y_train[np.ix_(sec_train, index)]
  x_val_in, x_val_out = x_val[sec_val, :], x_val[~sec_val, :]
  y_val_in = y_val[np.ix_(sec_val, index)]
  x_test_in, x_test_out = x_test[sec_test, :], x_test[~sec_test, :]
  y_test_in = y_test[np.ix_(sec_test, index)]
  
  # Compute the features
  submodel = Model(inputs=model.input, 
                   outputs=model.get_layer('features_layer').output)
  submodel.load_weights(model_path, by_name=True)
  
  features_train_in = submodel.predict(x_train_in)
  features_train_out = submodel.predict(x_train_out)
  features_val_in = submodel.predict(x_val_in)
  features_val_out = submodel.predict(x_val_out)
  features_test_in = submodel.predict(x_test_in)
  features_test_out = submodel.predict(x_test_out)
  
  np.savez(os.path.join(path_dir, 'features.npz'), 
           features_train_in=features_train_in,
           features_train_out=features_train_out,
           features_val_in=features_val_in,
           features_val_out=features_val_out,
           features_test_in=features_test_in,
           features_test_out=features_test_out
          )
  
  np.savez(os.path.join(path_dir, 'y.npz'),
           y_train_in=y_train_in,
           y_val_in=y_val_in,
           y_test_in=y_test_in
          )


if __name__ == '__main__':
  app.run(main, argv=[100, 'first', 'cifar100'])

