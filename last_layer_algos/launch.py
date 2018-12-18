#%% Imports

import os
from absl import app
import itertools
import h5py

import numpy as np
import keras

import utils.sgld as sgld
import utils.util as util


#%% Algorithms

def input_data(hparams):
  
  input_path = 'saved_models/{}'.format(hparams['dataset'])
  features = np.load(os.path.join(input_path, 'features.npz'))
  y = np.load(os.path.join(input_path, 'y.npz'))

  features_train_in = features['features_train_in']
  y_train_in = y['y_train_in']
  features_val_in = features['features_val_in']
  y_val_in = y['y_val_in']
  features_val_out = features['features_val_out']
  
  if features_val_out.shape[0] == 0:
    features_val_out = None
    print('No out of distribution samples')

  return (features_train_in, y_train_in), (features_val_in, 
         y_val_in), features_val_out


def sgd_sgld(hparams):  
  
  output_dir = util.create_run_dir('outputs/last_layer/', hparams)
  util.write_to_csv(os.path.join(output_dir, 'hparams.csv'), hparams)
  
  (features_train_in, y_train_in), (features_val_in, y_val_in), \
    features_val_out = input_data(hparams)
    
  n_class = y_train_in.shape[1]
#  epochs = hparams['epochs']
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
    
  model = util.build_last_layer(features_train_in, n_class)
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

  model = util.build_last_layer(features_train_in, n_class)
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

  model = util.build_last_layer(features_train_in, n_class, 
                                p_dropout=p_dropout)
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
#  del argv
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
  
#  hparams = {'dataset': 'mnist-first-10',
#           'algorithm': 'dropout',
#           'epochs': 10,
#           'batch_size': 32,
#           'lr': 0.01,
#           'p_dropout': 0.5,
#           'samples': 10
#          }
  
  if algo == 'sgdsgld': 
    # 30 sim
    list_dataset = ['mnist-first-10']
    list_algorithms = ['sgdsgld']
    list_samples = [10, 100, 1000]
    list_batch_size = [32, 64]
    list_lr = [0.1, 0.05, 0.01, 0.005, 0.001]
  
    list_epochs = [100]
    list_p_dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
  elif algo == 'dropout':
    # 150
    list_dataset = ['mnist-first-10']
    list_algorithms = ['dropout']
    list_samples = [10, 100, 1000]
    list_batch_size = [32, 64]
    list_lr = [0.1, 0.05, 0.01, 0.005, 0.001]
  
    list_epochs = [100]
    list_p_dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
  elif algo == 'bootstrap':
    # 20
    list_dataset = ['mnist-first-10']
    list_algorithms = ['bootstrap']
    list_samples = [10, 100]
    list_batch_size = [32, 64]
    list_lr = [0.1, 0.05, 0.01, 0.005, 0.001]
  
    list_epochs = [10]
    list_p_dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
  else:
    raise ValueError('this algorithm is not supported')
  
  
  
  i = 0
  def smartprint(i):
    print('----------------------')
    print('End of {} step'.format(i))
    print('----------------------')
  
  for dataset, algorithm, samples, batch_size, lr in \
    itertools.product(list_dataset, list_algorithms,  
                      list_samples, list_batch_size, list_lr):
  
    hparams = {'dataset': dataset,
               'algorithm': algorithm,
               'epochs': 100,
               'batch_size': batch_size,
               'lr': lr,
               'p_dropout': 0.5,
               'samples': samples
              }
    
    if algorithm == 'bootstrap':
      for epochs in list_epochs:
        hparams['epochs'] = epochs
        bootstrap(hparams)
        i += 1
        smartprint(i)
    elif algorithm == 'dropout':
      for epochs, p_dropout in itertools.product(list_epochs, list_p_dropout):
        hparams['epochs'] = epochs
        hparams['p_dropout'] = p_dropout
        dropout(hparams)
        i += 1
        smartprint(i)
    elif algorithm == 'sgdsgld':
      sgd_sgld(hparams)
      i += 1
      smartprint(i)
    else:
      raise ValueError('this algorithm is not supported')
    

    
if __name__ == '__main__':
  app.run(main)  # mnist, cifar10, cifar100
  
  







