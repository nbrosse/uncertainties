#%% Imports

import os
from absl import app
import itertools
import h5py

import numpy as np
import keras
from keras.layers import Input, Dense
from keras.models import Sequential, Model

import utils.sgld as sgld
import utils.util as util
from utils.dropout_layer import PermaDropout


#%% Algorithms

def build_last_layer(features_train, num_classes, 
                     p_dropout=None):
  """Build the last layer keras model.
  
  Args:
    features_train: features of the trainig set.
    num_classes: int, number of classes.
    p_dropout: float between 0 and 1. Fraction of the input units to drop.
  Returns:
    submodel: last layer model.
  """
  n = features_train.shape[0]
  features_shape = (features_train.shape[1],)
  if p_dropout is not None:
    x = Input(shape=features_shape, name='ll_input')
    y = PermaDropout(p_dropout, name='ll_dropout')(x)
    y = Dense(num_classes, activation='softmax', name='ll_dense',
              kernel_regularizer=keras.regularizers.l2(1./n),
              bias_regularizer=keras.regularizers.l2(1./n))(y)
    model = Model(inputs=x, outputs=y)
  else:
    model = Sequential()
    model.add(Dense(num_classes, activation='softmax', 
                    input_shape=features_shape, name='ll_dense',
                    kernel_regularizer=keras.regularizers.l2(1./n),
                    bias_regularizer=keras.regularizers.l2(1./n)))
  return model


def input_data(hparams):
  
  input_path = 'saved_models/{}'.format(hparams['dataset'])
  features = np.load(os.path.join(input_path, 'features.npz'))
  y = np.load(os.path.join(input_path, 'y.npz'))
  
  features_train_in = features['features_train_in']
  y_train_in = y['y_train_in']
  y_val_in = y['y_test_in']
  features_val_out = features['features_test_out']
  features_val_in = features['features_test_in']
  
  if features_val_out.shape[0] == 0:
    features_val_out = None
    print('No out of distribution samples')

  return (features_train_in, y_train_in), (features_val_in, 
         y_val_in), features_val_out

def onepoint(hparams):
  
  output_dir = util.create_run_dir('outputs/last_layer/', hparams)
  
  (features_train_in, y_train_in), (features_val_in, y_val_in), \
    features_val_out = input_data(hparams)
    
  n_class = y_train_in.shape[1]
  
  model = build_last_layer(features_train_in, n_class)
  model_path = 'saved_models/{}/{}.h5'.format(hparams['dataset'], 
                             hparams['dataset'].split('-')[0])

  model.load_weights(model_path, by_name=True)
  
  name_in = os.path.join(output_dir, 'p_in.h5')
  file_in = h5py.File(name_in, 'a')
  shape_in = (features_val_in.shape[0], n_class, 1)
  proba_in = file_in.create_dataset('proba', 
                                    shape_in,
                                    # dtype='f2',
                                    compression='gzip')
  if features_val_out is not None:
    name_out = os.path.join(output_dir, 'p_out.h5')
    file_out = h5py.File(name_out, 'a')
    shape_out = (features_val_out.shape[0], n_class, 1)
    proba_out = file_out.create_dataset('proba', 
                                        shape_out,
                                        # dtype='f2',
                                        compression='gzip') 

  proba_in[:, :, 0] = model.predict(features_val_in)
  if features_val_out is not None:
    proba_out[:, :, 0] = model.predict(features_val_out)
    
  file_in.close()
  if features_val_out is not None:
    file_out.close()


def sgd_sgld(hparams):  
  
  output_dir = util.create_run_dir('outputs/last_layer/', hparams)
  util.write_to_csv(os.path.join(output_dir, 'hparams.csv'), hparams)
  
  (features_train_in, y_train_in), (features_val_in, y_val_in), \
    features_val_out = input_data(hparams)
    
  n_class = y_train_in.shape[1]
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
    
  model = build_last_layer(features_train_in, n_class)
  model_path = 'saved_models/{}/{}.h5'.format(hparams['dataset'], 
                             hparams['dataset'].split('-')[0])

  for opt, optimizer in zip(['sgd', 'sgld'], 
                            [keras.optimizers.SGD(lr=lr), 
                             sgld.SGLD(features_train_in.shape[0], lr=lr)]):
    model.load_weights(model_path, by_name=True)
    params['optimizer'] = opt
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    mc = Prediction(params, features_val_in, features_val_out)
  
    hist = model.fit(features_train_in, y_train_in,
                     batch_size=batch_size,
                     epochs=samples,
                     verbose=1,
                     validation_data=(features_val_in, y_val_in),
                     callbacks=[mc])
    print('End of sampling using {}'.format(opt))

  return hist


def bootstrap(hparams):
  
  output_dir = util.create_run_dir('outputs/last_layer/', hparams)
  util.write_to_csv(os.path.join(output_dir, 'hparams.csv'), hparams)
  
  (features_train_in, y_train_in), (features_val_in, y_val_in), \
    features_val_out = input_data(hparams)
    
  n_class = y_train_in.shape[1]
  epochs = hparams['epochs']
  lr = hparams['lr']
  batch_size = hparams['batch_size']
  samples = hparams['samples']

  model = build_last_layer(features_train_in, n_class)
  model_path = 'saved_models/{}/{}.h5'.format(hparams['dataset'], 
                             hparams['dataset'].split('-')[0])

  model.compile(optimizer=keras.optimizers.SGD(lr=lr),
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
  
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
    model.fit(bootstrap_features_train_in, bootstrap_y_train_in,
              batch_size=batch_size,
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
  
  output_dir = util.create_run_dir('outputs/last_layer/', hparams)
  util.write_to_csv(os.path.join(output_dir, 'hparams.csv'), hparams)
  
  (features_train_in, y_train_in), (features_val_in, y_val_in), \
    features_val_out = input_data(hparams)
    
  n_class = y_train_in.shape[1]
  epochs = hparams['epochs']
  lr = hparams['lr']
  batch_size = hparams['batch_size']
  samples = hparams['samples']
  p_dropout = hparams['p_dropout']

  model = build_last_layer(features_train_in, n_class, p_dropout=p_dropout)
  model_path = 'saved_models/{}/{}.h5'.format(hparams['dataset'], 
                             hparams['dataset'].split('-')[0])

  model.compile(optimizer=keras.optimizers.SGD(lr=lr),
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
  
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
  model.fit(features_train_in, y_train_in,
            batch_size=batch_size,
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

#%% Sample

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
             'algorithm': algo
            }
  
  list_batch_size = [128]
  list_lr = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]

  if algo == 'sgdsgld': 
    # 5 sim
    list_samples = [10, 100, 1000]
  elif algo == 'dropout':
    # 15
    list_samples = [10, 100, 1000]
    list_epochs = [100]
    list_p_dropout = [0.1, 0.3, 0.5]
  elif algo == 'bootstrap':
    # 5
    list_samples = [10, 100]
    list_epochs = [10]
  elif algo == 'onepoint':
    onepoint(hparams)
    return 
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
  
  







