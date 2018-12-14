#%% Imports

import os
from absl import app
import pickle

import numpy as np
import keras

import utils.sgld as sgld
import utils.util as util

import glob

import itertools

#%% Launch functions

"""
dataset: mnist, cifar10, cifar100, diabetic, imagenet, 
         mnist_{first, last, random}_{n_class}, etc.
algorithm: sgd_sgld, bootstrap, dropout
epochs: 10, 100, 1000, 10000
thinning_interval: 1
num_samples: epochs // thinning_interval
batch_size: 32, 64, 128
lr: 0.1, 0.05, 0.01, 0.005, 0.001
p_dropout: 0.2, 0.3, 0.4, 0.5 
"""

hparams = {'dataset': 'mnist',
           'algorithm': 'sgd_sgld',
           'num_classes': 10,
           'epochs': 10,
           'thinning_interval': 1,
           'batch_size': 32,
           'lr': 0.01,
           'num_samples': 10,
           'p_dropout': 0.5
          }

dataset = hparams['dataset']

saved_model_dir = 'saved_models/{}'.format(dataset)

def input_data(hparams):
  input_path='../saved_models/{}_first_{}'.format(hparams['dataset'], hparams['num_classes'])
  features_path=os.path.join(input_path, 'features.npz')
  y_path=os.path.join(input_path, 'y.npz')
  features=np.load(features_path)
  y=np.load(y_path)

  features_train_in=features[features.files[0]]
  y_train_in=y[y.files[0]]

  features_val_in = features[features.files[2]]
  y_val_in=y[y.files[1]]

  features_val_out = features[features.files[3]]
  if features_val_out.shape[0]==0:
    features_val_out=features_val_in
    print('no out_of_distribution samples')

  return (features_train_in, y_train_in), (features_val_in, y_val_in), features_val_out


def sgd_sgld_last_layer(model, optimizer, epochs, batch_size,
                        features_train, y_train, features_val_in, features_val_out, y_val_in):
  #thinning_interval, path_weights (old parameters)

  """Train last layer model using SGD and SGLD.
  Weights snapshots every thinning_interval.

  Args:
    model: keras model.
    optimizer: optimizer
    epochs: epochs
    batch_size: batch_size
    features_train: features_train of the last layer
    y_train: y_train
    features_test: features_test of the last layer
    y_test: y_test
    thinning_interval: int, thinning interval between snapshots.
    path_weights: str, directory where to write snapshots of the weights
  Returns:
    hist: history (keras) object of the training
  """

  # compile the model
  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  # Saving after every N batches
  #mc = keras.callbacks.ModelCheckpoint(path_weights+'_weights{epoch:03d}.h5', save_weights_only=True, period=thinning_interval)

  class prediction_history(keras.callbacks.Callback):

    def __init__(self, features_val_in, features_val_out):
      self.predhis_in = []
      self.predhis_out = []
      self.features_val_in = features_val_in
      self.features_val_out = features_val_out

    def on_epoch_end(self, epoch, logs={}):
      self.predhis_in.append(self.model.predict(self.features_val_in))
      self.predhis_out.append(self.model.predict(self.features_val_out))


  mc=prediction_history(features_val_in, features_val_out)

  hist = model.fit(features_train, y_train,
                   batch_size=batch_size,
                   epochs=epochs,
                   verbose=1,
                   validation_data=(features_val_in, y_val_in),
                   callbacks=[mc])

  return hist, mc.predhis_in, mc.predhis_out


def launch_sgd_sgld(hparams):
  output_dir = util.create_run_dir('../outputs/last_layer/', hparams)
  util.write_to_csv(os.path.join(output_dir, 'hparams.csv'), hparams)

  (features_train, y_train), (features_val_in, y_val_in), features_val_out = input_data(hparams)

  model = util.build_last_layer(features_train, hparams['num_classes'])
  model.load_weights('../saved_models/{}_first_{}/{}.h5'.format(hparams['dataset'], hparams['num_classes'], hparams['dataset']), by_name=True)

  # Sampling
  for opt, optimizer in zip(['sgd', 'sgld'], [keras.optimizers.SGD(lr=hparams['lr']), sgld.SGLD(features_train.shape[0], lr=hparams['lr'])]):
    #path_weights = os.path.join(output_dir, opt)
    hist, proba_tab_in, proba_tab_out=sgd_sgld_last_layer(model,optimizer, hparams['epochs'], hparams['batch_size'], features_train, y_train, features_val_in, features_val_out, y_val_in)
    np.save(os.path.join(output_dir, 'p_in_{}.npy'.format(opt)), proba_tab_in)
    np.save(os.path.join(output_dir, 'p_out_{}.npy'.format(opt)), proba_tab_out)

  #hist_sgd_sgld.append(hist)

  # Compute the probabilities
  #for opt in ['sgd', 'sgld']:
    #proba_tab_in, proba_tab_out= predict_sgd_sgld_last_layer(model, features_val_in, features_val_out, hparams['num_classes'], opt)
    #np.save(os.path.join(output_dir,'p_in_{}.npy'.format(opt)), proba_tab_in)
    #np.save(os.path.join(output_dir, 'p_out_{}.npy'.format(opt)), proba_tab_out)
  print('End of computing probabilities')


def launch_bootstrap(hparams, num_samples=10):
  output_dir = util.create_run_dir('../outputs/last_layer/', hparams)
  util.write_to_csv(os.path.join(output_dir, 'hparams.csv'), hparams)

  (features_train, y_train), (features_val_in, y_val_in), features_val_out = input_data(hparams)

  model = util.build_last_layer(features_train, hparams['num_classes'])
  model.load_weights('../saved_models/{}_first_{}/{}.h5'.format(hparams['dataset'], hparams['num_classes'], hparams['dataset']), by_name=True)

  model.compile(optimizer=keras.optimizers.SGD(lr=hparams['lr']),
                   loss='categorical_crossentropy', 
                   metrics=['accuracy'])

  # Bootstrap
  proba_tab_in = np.zeros((features_val_in.shape[0], hparams['num_classes'], num_samples))
  proba_tab_out = np.zeros((features_val_out.shape[0], hparams['num_classes'], num_samples))
  for i in np.arange(num_samples):
    bootstrap_features_train, bootstrap_y_train = util.bootstrap(features_train, y_train)
    model.fit(bootstrap_features_train, bootstrap_y_train,
              batch_size=hparams['batch_size'],
              epochs=hparams['epochs'],
              verbose=1,
              validation_data=(features_val_in, y_val_in))
    # Sanity check
    score = model.evaluate(features_val_in, y_val_in, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('End of boostrap {}'.format(i))

    # computing probabilities
    proba_tab_in[:, :, i] = model.predict(features_val_in)
    proba_tab_out[:, :, i] = model.predict(features_val_out)
  print('End of sampling.')
  np.save(os.path.join(output_dir, 'p_in_bootstrap.npy'), proba_tab_in)
  np.save(os.path.join(output_dir, 'p_out_bootstrap.npy'), proba_tab_out)

  print('End of computing probabilities')


def launch_dropout(hparams, num_samples=10):
  output_dir = util.create_run_dir('../outputs/last_layer/', hparams)
  util.write_to_csv(os.path.join(output_dir, 'hparams.csv'), hparams)

  (features_train, y_train), (features_val_in, y_val_in), features_val_out = input_data(hparams)

  model = util.build_last_layer(features_train, hparams['num_classes'])
  model.load_weights('../saved_models/{}_first_{}/{}.h5'.format(hparams['dataset'], hparams['num_classes'], hparams['dataset']),by_name=True)


  model.compile(optimizer=keras.optimizers.SGD(lr=hparams['lr']),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  # Training
  hist = model.fit(features_train, y_train,
                      batch_size=hparams['batch_size'],
                      epochs=hparams['epochs'],
                      verbose=1,
                      validation_data=(features_val_in, y_val_in))
  # Saving the model
  model.save_weights(os.path.join(output_dir, 'weights.h5'))

  # Save history of the training
  with open(os.path.join(output_dir, 'metrics_hist_dropout.pkl'), 'wb') as handle:
      pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

  # Sampling
  proba_tab_in = np.zeros(shape=(features_val_in.shape[0], hparams['num_classes'], num_samples))
  proba_tab_out= np.zeros(shape=(features_val_out.shape[0], hparams['num_classes'], num_samples))
  for index in np.arange(num_samples):
    proba_in = model.predict(features_val_in)
    proba_out= model.predict(features_val_out)
    proba_tab_in[:, :, index] = proba_in
    proba_tab_out[:, :, index] = proba_out

  # Saving the probabilities
  np.save(os.path.join(output_dir, 'p_in_dropout.npy'), proba_tab_in)
  np.save(os.path.join(output_dir, 'p_out_dropout.npy'), proba_tab_in)

#%%#Sample

def main(argv):
  del argv
  if hparams['algorithm']=='sgd_sgld':
    launch_sgd_sgld(hparams)
  elif hparams['algorithm']=='bootstrap':
    launch_bootstrap(hparams)
  elif hparams['algorithm'] == 'dropout':
    launch_dropout(hparams)
  else:
    raise ValueError('this algorithm is not supported')

    
if __name__ == '__main__':
  app.run(main)  # mnist, cifar10, cifar100
  
  
