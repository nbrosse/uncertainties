"""Specific version of launch for ImageNet using tfrecords.
Work in progress.
"""

#%% Imports

import os
import gc
import glob
from absl import app
import itertools
import h5py

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, Model
import keras

#from utils.dropout_layer import PermaDropout
#import utils.sgld as sgld
import utils.tf_sgld as tf_sgld
import utils.util as util


#%% Useful constants for Imagenet

DIM = 4032
NUM_TRAINING_EXAMPLES = 1281167
NUM_TEST_EXAMPLES = 50000
N_CLASS = 1000

#%% Algorithms

def input_data(hparams):
  features_dir_train = '/home/nbrosse/features_imagenet/train/train-*'
  features_dir_val = '/home/nbrosse/features_imagenet/test/test-*'
  
  batch_size = hparams['batch_size']
  
  file_list_train = glob.glob(features_dir_train) 
  file_list_val = glob.glob(features_dir_val) 
  
  def parse_proto(example_proto, d=DIM):
    features = {
      'X': tf.FixedLenFeature([d], tf.float32),
      'y': tf.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features['X'], tf.one_hot(parsed_features['y'], 
                          depth=N_CLASS)
  
  def read_tfrecords(file_names,
                     is_training,
                     buffer_size=10000,
                     batch_size=batch_size):
    dataset = tf.data.TFRecordDataset(file_names)
    dataset = dataset.map(parse_proto)
    if is_training:
      dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat()
  
    return dataset
  
  dataset_train = read_tfrecords(file_names=file_list_train, is_training=True)
  dataset_val = read_tfrecords(file_names=file_list_val, is_training=False)
  
  return dataset_train, dataset_val

def build_last_layer(n, d, num_classes, p_dropout=None):
  """Build the last layer keras model.
  
  Args:
    features_train: features of the trainig set.
    num_classes: int, number of classes.
    p_dropout: float between 0 and 1. Fraction of the input units to drop.
  Returns:
    submodel: last layer model.
  """
  features_shape = (d,)
  if p_dropout is not None:
    x = Input(shape=features_shape, name='ll_input')
#    y = PermaDropout(p_dropout, name='ll_dropout')(x)
    y = tf.layers.dropout(x, rate=p_dropout, training=True, name='ll_dropout')
    y = Dense(num_classes, activation='softmax', name='ll_dense',
              kernel_regularizer=tf.keras.regularizers.l2(1./n),
              bias_regularizer=tf.keras.regularizers.l2(1./n))(y)
    model = Model(inputs=x, outputs=y)
  else:
    model = Sequential()
    model.add(Dense(num_classes, activation='softmax', 
                    input_shape=features_shape, name='ll_dense',
                    kernel_regularizer=tf.keras.regularizers.l2(1./n),
                    bias_regularizer=tf.keras.regularizers.l2(1./n)))
  return model

config_cpu = tf.ConfigProto(device_count = {'GPU': 0})

def compute_y(hparams):
  _, dataset_val = input_data(hparams)
  batch_size = hparams['batch_size']
  iterator = dataset_val.make_one_shot_iterator()
  next_element = iterator.get_next()
  y_aggr = np.zeros(((NUM_TEST_EXAMPLES // batch_size) * batch_size, 1000))
  with tf.Session(config=config_cpu) as sess:
    for i in np.arange(NUM_TEST_EXAMPLES // batch_size):
        _, y = sess.run(next_element)
        y_aggr[i*batch_size:(i+1)*batch_size] = y

  return y_aggr

def sgd_sgld(hparams, only_sgld=True):  
  
  output_dir = util.create_run_dir('outputs/last_layer/', hparams)
  util.write_to_csv(os.path.join(output_dir, 'hparams.csv'), hparams)
  
  dataset_train, dataset_val = input_data(hparams)
    
  n_class = N_CLASS
  samples = hparams['samples']
  lr = hparams['lr']
  batch_size = hparams['batch_size']

  params = {'optimizer': None,
            'samples': samples,
            'batch_size': batch_size,
            'output_dir': output_dir,
            'n_class': n_class
            }

  class Prediction(keras.callbacks.Callback):

    def __init__(self, params, dataset_val):
      super(Prediction, self).__init__()
      
      self.index = 0
      
      name_in = os.path.join(params['output_dir'], 
                             'p_{}_in.h5'.format(params['optimizer']))
      self.file_in = h5py.File(name_in, 'a')
      
      self.batch_size = params['batch_size']
      
      shape_in = ((NUM_TEST_EXAMPLES // self.batch_size) * self.batch_size,
                  params['n_class'], 
                  params['samples'])
      
      self.proba_in = self.file_in.create_dataset('proba', 
                                                  shape_in,
                                                  # dtype='f2',
                                                  compression='gzip')
      self.dataset_val = dataset_val

    def on_epoch_end(self, epoch, logs={}):
      nb_steps = NUM_TEST_EXAMPLES // self.batch_size 
      self.proba_in[:, :, self.index] = self.model.predict(self.dataset_val, 
                   steps=nb_steps)
      self.index += 1
      
    def on_train_end(self, logs={}):
      self.file_in.close()
    
  model = build_last_layer(n=NUM_TRAINING_EXAMPLES, 
                           d=DIM, num_classes=N_CLASS)
  model_path = 'saved_models/{}/{}.h5'.format(hparams['dataset'], 
                             hparams['dataset'].split('-')[0])

  for opt, optimizer in zip(['sgd', 'sgld'], 
                            [tf.keras.optimizers.SGD(lr=lr), 
                             tf_sgld.SGLD(NUM_TRAINING_EXAMPLES, lr=lr)]):
    if (opt == 'sgd') & only_sgld:
      continue
    model.load_weights(model_path, by_name=True)
    params['optimizer'] = opt
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    mc = Prediction(params, dataset_val)
  
    hist = model.fit(dataset_train,
                     epochs=samples,
                     steps_per_epoch=NUM_TRAINING_EXAMPLES // batch_size,
                     verbose=1,
                     validation_data=dataset_val,
                     validation_steps=NUM_TEST_EXAMPLES // batch_size,
                     callbacks=[mc])
    print('End of sampling using {}'.format(opt))
  
  del model 
  gc.collect()
  return hist


def dropout(hparams):
  
  output_dir = util.create_run_dir('outputs/last_layer/', hparams)
  util.write_to_csv(os.path.join(output_dir, 'hparams.csv'), hparams)
  
  dataset_train, dataset_val = input_data(hparams)
    
  n_class = N_CLASS
  epochs = hparams['epochs']
  lr = hparams['lr']
  batch_size = hparams['batch_size']
  samples = hparams['samples']
  p_dropout = hparams['p_dropout']

  model = build_last_layer(n=NUM_TRAINING_EXAMPLES, 
                           d=DIM, 
                           num_classes=n_class,
                           p_dropout=p_dropout)
  model_path = 'saved_models/{}/{}.h5'.format(hparams['dataset'], 
                             hparams['dataset'].split('-')[0])

  model.compile(optimizer=tf.keras.optimizers.SGD(lr=lr),
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
  
  name_in = os.path.join(output_dir, 'p_in.h5')
  file_in = h5py.File(name_in, 'a')
  
  shape_in = ((NUM_TEST_EXAMPLES // batch_size) * batch_size,
              N_CLASS, 
              samples)
  
  proba_in = file_in.create_dataset('proba', 
                                    shape_in,
                                    # dtype='f2',
                                    compression='gzip')     

  model.load_weights(model_path, by_name=True)
  hist = model.fit(dataset_train,
                   epochs=epochs,
                   steps_per_epoch=NUM_TRAINING_EXAMPLES // batch_size,
                   verbose=1,
                   validation_data=dataset_val,
                   validation_steps=NUM_TEST_EXAMPLES // batch_size
                  )
  print('End of training')

  for i in np.arange(samples):
    # computing probabilities
    proba_in[:, :, i] = model.predict(dataset_val,
            steps=NUM_TEST_EXAMPLES // batch_size)
  
  file_in.close()
  del model
  gc.collect()
  print('End of sampling - dropout.')

#%% Sample

def main(argv):
#  del argv
  algorithm = 'dropout' # argv[1]
  batch_size = 512 # int(argv[2])
  
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
  
  dataset = 'imagenet-first-1000'
  
#  hparams = {'dataset': 'mnist-first-10',
#             'algorithm': 'sgdsgld',
#             'epochs': 10,
#             'batch_size': 64,
#             'lr': 0.001,
#             'p_dropout': 0.5,
#             'samples': 1000
#            }
#  
#  sgd_sgld(hparams)
  
  if algorithm == 'sgdsgld': 
    # 4 sim
    list_samples = [100]
    list_lr = [0.01]
  
#    list_epochs = [100]
#    list_p_dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
  elif algorithm == 'dropout':
    list_samples = [100]
    list_lr = [0.01]
  
    list_epochs = [100]
    list_p_dropout = [0.1]
#  elif algorithm == 'bootstrap':
#    # 10
#    list_dataset = ['cifar100-first-100']
#    list_algorithms = ['bootstrap']
#    list_samples = [10, 100]
#    list_batch_size = [128]
#    list_lr = [0.001, 0.005, 0.0001, 0.00005, 0.00001]
#  
#    list_epochs = [10]
#    list_p_dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
  else:
    raise ValueError('this algorithm is not supported')

  hparams = {'dataset': dataset,
             'algorithm': algorithm,
             'batch_size': batch_size
            }
  
#  y_aggr = compute_y(hparams)
#  
#  np.save('y.npy', y_aggr)
  
  i = 0
  def smartprint(i):
    print('----------------------')
    print('End of {} step'.format(i))
    print('----------------------')
  
  for samples, lr in itertools.product(list_samples, list_lr):
  
    hparams = {'dataset': dataset,
               'algorithm': algorithm,
               'epochs': 100,
               'batch_size': batch_size,
               'lr': lr,
               'p_dropout': 0.1,
               'samples': samples
              }
    
#    if algorithm == 'bootstrap':
#      for epochs in list_epochs:
#        hparams['epochs'] = epochs
#        bootstrap(hparams)
#        i += 1
#        smartprint(i)
    if algorithm == 'dropout':
      for epochs, p_dropout in itertools.product(list_epochs, list_p_dropout):
        hparams['epochs'] = epochs
        hparams['p_dropout'] = p_dropout
        dropout(hparams)
        i += 1
        smartprint(i)
    if algorithm == 'sgdsgld':
      sgd_sgld(hparams)
      i += 1
      smartprint(i)
#    else:
#      raise ValueError('this algorithm is not supported')
    

    
if __name__ == '__main__':
  app.run(main)  # mnist, cifar10, cifar100
  
  







