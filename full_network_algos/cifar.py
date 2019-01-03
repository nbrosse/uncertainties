
#%% Packages

from __future__ import print_function

from absl import app
import os
import shutil
import itertools
import h5py

import numpy as np

import keras
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras import regularizers

import utils.util as util
import utils.sgld as sgld
from utils.dropout_layer import PermaDropout

WEIGHT_DECAY = 0.0005
SHAPE = [32, 32, 3]

#%% Model

def input_data(hparams):
  dataset = hparams['dataset']
  n_class = hparams['n_class']
  method = hparams['method']
  if dataset.startswith('cifar10'):
    (x_train, y_train),(x_test, y_test) = cifar10.load_data()
  else:
    (x_train, y_train),(x_test, y_test) = cifar100.load_data()
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  y_train = keras.utils.to_categorical(y_train, n_class)
  y_test = keras.utils.to_categorical(y_test, n_class)
  
  x_train, (mean, std) = whitening(x_train)
  x_test = normalize(x_test, mean, std)
  
  index = util.select_classes(y_train, n_class, method=method)
  sec_train = np.dot(y_train, index).astype(bool)
  sec_test = np.dot(y_test, index).astype(bool)
  
  x_train_in = x_train[sec_train, :]
  y_train_in = y_train[np.ix_(sec_train, index)]
  x_test_in, x_test_out = x_test[sec_test, :], x_test[~sec_test, :]
  y_test_in = y_test[np.ix_(sec_test, index)]
  
  return (x_train_in, y_train_in), (x_test_in, y_test_in), x_test_out, index

def build_model(n_class, p_dropout=None):
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
  if p_dropout is None:
    model.add(Dropout(0.3))
  else:
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
  if p_dropout is None:
    model.add(Dropout(0.4))
  else:
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
  if p_dropout is None:
    model.add(Dropout(0.4))
  else:
    model.add(PermaDropout(p_dropout))

  model.add(Conv2D(256, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  if p_dropout is None:
    model.add(Dropout(0.4))
  else:
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
  if p_dropout is None:
    model.add(Dropout(0.4))
  else:
    model.add(PermaDropout(p_dropout))

  model.add(Conv2D(512, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  if p_dropout is None:
    model.add(Dropout(0.4))
  else:
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
  if p_dropout is None:
    model.add(Dropout(0.4))
  else:
    model.add(PermaDropout(p_dropout))

  model.add(Conv2D(512, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  if p_dropout is None:
    model.add(Dropout(0.4))
  else:
    model.add(PermaDropout(p_dropout))

  model.add(Conv2D(512, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(MaxPooling2D(pool_size=(2, 2)))
  if p_dropout is None:
    model.add(Dropout(0.5))
  else:
    model.add(PermaDropout(p_dropout))

  model.add(Flatten())
  model.add(Dense(512,kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
  model.add(Activation('relu'))
  model.add(BatchNormalization(name='features_layer'))

  if p_dropout is None:
    model.add(Dropout(0.5))
  else:
    model.add(PermaDropout(p_dropout))
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


def sgd_sgld(hparams):  
  
  n_class = hparams['n_class']
  
  output_dir = util.create_run_dir('outputs/full_network/', hparams)
  util.write_to_csv(os.path.join(output_dir, 'hparams.csv'), hparams)
    
  (features_train_in, y_train_in), (features_val_in, y_val_in), \
    features_val_out, index = input_data(hparams)
    
  np.save(os.path.join(output_dir, 'index.npy'), index)

  samples = hparams['samples']
  lr = hparams['lr']
  batch_size = hparams['batch_size']

  params = {'optimizer': None,
            'samples': samples,
            'output_dir': output_dir,
            'n_class': n_class
            }

  class Prediction(keras.callbacks.Callback):

    def __init__(self, params, features_val_in, features_val_out):
      super(Prediction, self).__init__()
      
      self.index = 0
      
      if features_val_out is None:
        self.out_of_dist = True
      else:
        self.out_of_dist = False
      
      name_in = os.path.join(params['output_dir'], 
                             'p_{}_in.h5'.format(params['optimizer']))
      self.file_in = h5py.File(name_in, 'a')
      
      shape_in = (features_val_in.shape[0], params['n_class'], 
                  params['samples'])
      
      self.proba_in = self.file_in.create_dataset('proba', 
                                                  shape_in,
                                                  # dtype='f2',
                                                  compression='gzip')
      self.features_val_in = features_val_in

      if not self.out_of_dist:
        name_out = os.path.join(params['output_dir'], 
                               'p_{}_out.h5'.format(params['optimizer']))
        self.file_out = h5py.File(name_out, 'a')
        shape_out = (features_val_out.shape[0], params['n_class'], 
                     params['samples'])
        self.proba_out = self.file_out.create_dataset('proba', 
                                                      shape_out,
                                                      # dtype='f2',
                                                      compression='gzip')      
        self.features_val_out = features_val_out

    def on_epoch_end(self, epoch, logs={}):
      self.proba_in[:, :, self.index] = self.model.predict(self.features_val_in)
      if not self.out_of_dist:
        self.proba_out[:, :, self.index] = self.model.predict(self.features_val_out)
      self.index += 1
      
    def on_train_end(self, logs={}):
      self.file_in.close()
      if not self.out_of_dist:
        self.file_out.close()
    
  model = build_model(n_class)
  model_path = 'saved_models/cifar-full-network/{}vgg.h5'.format(hparams['dataset'].split('-')[0])
  
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
  datagen.fit(features_train_in)

  for opt, optimizer in zip(['sgd', 'sgld'], 
                            [keras.optimizers.SGD(lr=lr), 
                             sgld.SGLD(features_train_in.shape[0], lr=lr)]):
    model.load_weights(model_path, by_name=True)
    params['optimizer'] = opt
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    mc = Prediction(params, features_val_in, features_val_out)
  
    hist = model.fit_generator(datagen.flow(features_train_in, y_train_in, 
                                            batch_size=batch_size),
                     steps_per_epoch=features_train_in.shape[0] // batch_size,
                     epochs=samples,
                     verbose=1,
                     validation_data=(features_val_in, y_val_in),
                     callbacks=[mc])
    
    print('End of sampling using {}'.format(opt))

  return hist


def bootstrap(hparams):
  
  n_class = hparams['n_class']
  
  output_dir = util.create_run_dir('outputs/full_network/', hparams)
  util.write_to_csv(os.path.join(output_dir, 'hparams.csv'), hparams)

  (features_train_in, y_train_in), (features_val_in, y_val_in), \
    features_val_out, index = input_data(hparams)
    
  np.save(os.path.join(output_dir, 'index.npy'), index)
  
  epochs = hparams['epochs']
  lr = hparams['lr']
  batch_size = hparams['batch_size']
  samples = hparams['samples']

  model = build_model(n_class)
  model_path = 'saved_models/cifar-full-network/{}vgg.h5'.format(hparams['dataset'].split('-')[0])

  model.compile(optimizer=keras.optimizers.SGD(lr=lr),
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
  
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
  datagen.fit(features_train_in)
  
  name_in = os.path.join(output_dir, 'p_in.h5')
  file_in = h5py.File(name_in, 'a')
  
  shape_in = (features_val_in.shape[0], n_class, samples)
  
  proba_in = file_in.create_dataset('proba', 
                                    shape_in,
                                    # dtype='f2',
                                    compression='gzip')

  if features_val_out is not None:
    name_out = os.path.join(output_dir, 'p_out.h5')
    file_out = h5py.File(name_out, 'a')
    shape_out = (features_val_out.shape[0], n_class, samples)
    proba_out = file_out.create_dataset('proba', 
                                        shape_out,
                                        # dtype='f2',
                                        compression='gzip')      

  for i in np.arange(samples):
    bootstrap_features_train_in, bootstrap_y_train_in = \
      util.bootstrap(features_train_in, y_train_in)
    model.load_weights(model_path, by_name=True)
    model.fit_generator(datagen.flow(bootstrap_features_train_in, 
                                     bootstrap_y_train_in,
                                     batch_size=batch_size),
              steps_per_epoch=bootstrap_features_train_in.shape[0] // batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(features_val_in, y_val_in))
    # Sanity check
    score = model.evaluate(features_val_in, y_val_in, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('End of boostrap {}'.format(i))

    # computing probabilities
    proba_in[:, :, i] = model.predict(features_val_in)
    if features_val_out is not None:
      proba_out[:, :, i] = model.predict(features_val_out)
  
  file_in.close()
  if features_val_out is not None:
    file_out.close()
  print('End of sampling - bootstrap.')


def dropout(hparams):
  
  n_class = hparams['n_class']
  
  output_dir = util.create_run_dir('outputs/full_network/', hparams)
  util.write_to_csv(os.path.join(output_dir, 'hparams.csv'), hparams)
    
  (features_train_in, y_train_in), (features_val_in, y_val_in), \
    features_val_out, index = input_data(hparams)
    
  np.save(os.path.join(output_dir, 'index.npy'), index)
  
  epochs = hparams['epochs']
  lr = hparams['lr']
  batch_size = hparams['batch_size']
  samples = hparams['samples']
  p_dropout = hparams['p_dropout']

  model = build_model(n_class, p_dropout=p_dropout)
  model_path = 'saved_models/cifar-full-network/{}vgg.h5'.format(hparams['dataset'].split('-')[0])

  model.compile(optimizer=keras.optimizers.SGD(lr=lr),
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
  
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
  datagen.fit(features_train_in)
  
  name_in = os.path.join(output_dir, 'p_in.h5')
  file_in = h5py.File(name_in, 'a')
  
  shape_in = (features_val_in.shape[0], n_class, samples)
  
  proba_in = file_in.create_dataset('proba', 
                                    shape_in,
                                    # dtype='f2',
                                    compression='gzip')

  if features_val_out is not None:
    name_out = os.path.join(output_dir, 'p_out.h5')
    file_out = h5py.File(name_out, 'a')
    shape_out = (features_val_out.shape[0], n_class, samples)
    proba_out = file_out.create_dataset('proba', 
                                        shape_out,
                                        # dtype='f2',
                                        compression='gzip')      

  model.load_weights(model_path, by_name=True)
  model.fit_generator(datagen.flow(features_train_in, y_train_in,
                                   batch_size=batch_size),
            steps_per_epoch=features_train_in.shape[0] // batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(features_val_in, y_val_in))
  # Sanity check
  score = model.evaluate(features_val_in, y_val_in, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
  print('End of training')

  for i in np.arange(samples):
    # computing probabilities
    proba_in[:, :, i] = model.predict(features_val_in)
    if features_val_out is not None:
      proba_out[:, :, i] = model.predict(features_val_out)
  
  file_in.close()
  if features_val_out is not None:
    file_out.close()
  print('End of sampling - dropout.')
  
#%% main function

def main(argv):
  algo = argv[1]
  
  # Hyperparameters
  """
  dataset: mnist, cifar10, cifar100, diabetic, imagenet under the form 
           mnist-{first, last, random}-{n_class}, etc.
  algorithm: sgdsgld, bootstrap, dropout
  epochs: 10, 100, 1000, 10000 (= samples for dropout and bootstrap)
  batch_size: 32, 64, 128
  lr: 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001
  p_dropout: 0.2, 0.3, 0.4, 0.5 
  """
  
  hparams = {'dataset': 'cifar10-first-10',
             'algorithm': algo,
             'n_class': 10,
             'method': 'first'
            }
  
  list_batch_size = [128]
  list_lr = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
  
  if algo == 'sgdsgld': 
    list_samples = [10, 100, 1000]
  elif algo == 'dropout':
    list_samples = [10, 100, 1000]
    list_epochs = [100]
    list_p_dropout = [0.1, 0.3, 0.5]
  elif algo == 'bootstrap':
    list_samples = [10, 100]
    list_epochs = [10]
  else:
    raise ValueError('this algorithm is not supported')
  
  i = 0
  def smartprint(i):
    print('----------------------')
    print('End of {} step'.format(i))
    print('----------------------')
  
  for samples, batch_size, lr in \
    itertools.product(list_samples, list_batch_size, list_lr):
  
    hparams['batch_size'] = batch_size
    hparams['lr'] = lr
    hparams['samples'] = samples
    # Technical reason.
    hparams['epochs'] = 10
    hparams['p_dropout'] = 0.5
    
    if algo == 'bootstrap':
      for epochs in list_epochs:
        hparams['epochs'] = epochs
        bootstrap(hparams)
        i += 1
        smartprint(i)
    elif algo == 'dropout':
      for epochs, p_dropout in itertools.product(list_epochs, list_p_dropout):
        hparams['epochs'] = epochs
        hparams['p_dropout'] = p_dropout
        dropout(hparams)
        i += 1
        smartprint(i)
    elif algo == 'sgdsgld':
      sgd_sgld(hparams)
      i += 1
      smartprint(i)
    else:
      raise ValueError('this algorithm is not supported')
    

    
if __name__ == '__main__':
  app.run(main)  # mnist, cifar10, cifar100

