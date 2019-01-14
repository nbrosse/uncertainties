"""Specific version of launch for ImageNet using tfrecords.
"""

#%% Imports

import os
import gc
import glob
from absl import app
#import itertools
import h5py

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import keras

import utils.tf_sgld as tf_sgld
import utils.util as util


#%% Useful constants for Imagenet

DIM = 4032
NUM_TRAINING_EXAMPLES = 1281167
NUM_TEST_EXAMPLES = 50000
N_CLASS = 1000

# https://stackoverflow.com/questions/6773584/how-is-pythons-glob-glob-ordered

#%% Algorithms

def input_data(hparams, bootstrap=False):
  features_dir_train = '/home/nbrosse/features_imagenet_nasnet/train/train-*'
  features_dir_val = '/home/nbrosse/features_imagenet_nasnet/test/test-*'
  
  batch_size = hparams['batch_size']
  
  file_list_train = sorted(glob.glob(features_dir_train)) 
  file_list_val = sorted(glob.glob(features_dir_val)) 
  
  if bootstrap:
    file_list_train = list(np.random.choice(file_list_train, 
                                            len(file_list_train)))
  
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

def build_last_layer(n=NUM_TRAINING_EXAMPLES, d=DIM, num_classes=N_CLASS, 
                     p_dropout=None, num_last_layers=1):
  """Build the last layer keras model.
  """
  features_shape = (d,)
  if p_dropout is not None:
    x = Input(shape=features_shape, name='ll_input')
    y = x
    for i in np.arange(num_last_layers, 0, -1, dtype=int):
      y = tf.layers.dropout(y, rate=p_dropout, training=True, 
                            name='ll_dropout_{}'.format(i))
      if i > 1:
        y = Dense(d, activation='relu', name='ll_dense_{}'.format(i),
                  kernel_regularizer=tf.keras.regularizers.l2(1./n),
                  bias_regularizer=tf.keras.regularizers.l2(1./n))(y)
      else:
        y = Dense(num_classes, activation='softmax', 
                  name='ll_dense_{}'.format(i),
                  kernel_regularizer=tf.keras.regularizers.l2(1./n),
                  bias_regularizer=tf.keras.regularizers.l2(1./n))(y)
    model = Model(inputs=x, outputs=y)
  else:
    x = Input(shape=features_shape, name='ll_input')
    y = x
    for i in np.arange(num_last_layers, 0, -1, dtype=int):
      if i > 1:
        y = Dense(d, activation='relu', name='ll_dense_{}'.format(i),
                  kernel_regularizer=tf.keras.regularizers.l2(1./n),
                  bias_regularizer=tf.keras.regularizers.l2(1./n))(y)
      else:
        y = Dense(num_classes, activation='softmax', 
                  name='ll_dense_{}'.format(i),
                  kernel_regularizer=tf.keras.regularizers.l2(1./n),
                  bias_regularizer=tf.keras.regularizers.l2(1./n))(y)
    model = Model(inputs=x, outputs=y)
  
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

def initial_training(hparams):
  
  dataset_train, dataset_val = input_data(hparams)
    
  samples = hparams['samples']
  batch_size = hparams['batch_size']
  nb_last_layers = hparams['nb_last_layers']

  model_path = 'saved_models/{}/{}_last_layer_{}.h5'.format(hparams['dataset'], 
                             hparams['dataset'].split('-')[0], nb_last_layers)

  modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(model_path, 
                                                       monitor='val_acc',
                                                       save_best_only=True,
                                                       save_weights_only=True)

  
  model = build_last_layer(num_last_layers=nb_last_layers)
  
  optimizer = tf.keras.optimizers.Adam()

  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  
  hist = model.fit(dataset_train,
                   epochs=samples,
                   steps_per_epoch=NUM_TRAINING_EXAMPLES // batch_size,
                   verbose=1,
                   validation_data=dataset_val,
                   validation_steps=NUM_TEST_EXAMPLES // batch_size,
                   callbacks=[modelcheckpoint])
  
  print('End of Training.')
  
  del model 
  gc.collect()
  return hist


def onepoint(hparams):
  
  output_dir = util.create_run_dir('outputs/last_layer/', hparams)
  
  dataset_train, dataset_val = input_data(hparams)
    
  n_class = N_CLASS
  lr = hparams['lr']
  batch_size = hparams['batch_size']
  nb_last_layers = hparams['nb_last_layers']
  
  model = build_last_layer(num_last_layers=nb_last_layers)
  model_path = 'saved_models/{}/{}_last_layer_{}.h5'.format(hparams['dataset'], 
                             hparams['dataset'].split('-')[0], nb_last_layers)
  
  # Useless in normal situation but needed here for technical reason.
  model.compile(optimizer=tf.keras.optimizers.SGD(lr=lr),
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

  model.load_weights(model_path, by_name=True)
  
  name_in = os.path.join(output_dir, 'p_in.h5')
  file_in = h5py.File(name_in, 'a')
  shape_in = ((NUM_TEST_EXAMPLES // batch_size) * batch_size,
              n_class, 
              1)
  proba_in = file_in.create_dataset('proba', 
                                    shape_in,
                                    # dtype='f2',
                                    compression='gzip')

  proba_in[:, :, 0] = model.predict(dataset_val, 
                                    steps=NUM_TEST_EXAMPLES // batch_size)
    
  file_in.close()
  del model
  gc.collect()


def sgd_sgld(hparams):  
  
  output_dir = util.create_run_dir('outputs/last_layer/', hparams)
  util.write_to_csv(os.path.join(output_dir, 'hparams.csv'), hparams)
  
  dataset_train, dataset_val = input_data(hparams)
    
  n_class = N_CLASS
  samples = hparams['samples']
  lr = hparams['lr']
  batch_size = hparams['batch_size']
  nb_last_layers = hparams['nb_last_layers']

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
    
  model = build_last_layer(num_last_layers=nb_last_layers)
  model_path = 'saved_models/{}/{}_last_layer_{}.h5'.format(hparams['dataset'], 
                             hparams['dataset'].split('-')[0], nb_last_layers)

  for opt, optimizer in zip(['sgd', 'sgld'], 
                            [tf.keras.optimizers.SGD(lr=lr), 
                             tf_sgld.SGLD(NUM_TRAINING_EXAMPLES, lr=lr)]):
#    if (opt == 'sgd') & only_sgld:
#      continue
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


def bootstrap(hparams):
  
  output_dir = util.create_run_dir('outputs/last_layer/', hparams)
  util.write_to_csv(os.path.join(output_dir, 'hparams.csv'), hparams)
  
  dataset_train, dataset_val = input_data(hparams)
    
  epochs = hparams['epochs']
  lr = hparams['lr']
  batch_size = hparams['batch_size']
  samples = hparams['samples']
  p_dropout = hparams['p_dropout']
  nb_last_layers = hparams['nb_last_layers']

  model = build_last_layer(p_dropout=p_dropout,
                           num_last_layers=nb_last_layers)
  
  model_path = 'saved_models/{}/{}_last_layer_{}.h5'.format(hparams['dataset'], 
                             hparams['dataset'].split('-')[0], nb_last_layers)

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

  for i in np.arange(samples):
    dataset_train, dataset_val = input_data(hparams, bootstrap=True)
    model.load_weights(model_path, by_name=True)
    hist = model.fit(dataset_train,
                     epochs=epochs,
                     steps_per_epoch=NUM_TRAINING_EXAMPLES // batch_size,
                     verbose=1,
                     validation_data=dataset_val,
                     validation_steps=NUM_TEST_EXAMPLES // batch_size
                     )
    print('End of boostrap {}'.format(i))

    # computing probabilities
    proba_in[:, :, i] = model.predict(dataset_val,
            steps=NUM_TEST_EXAMPLES // batch_size)
  
  file_in.close()
  del model
  gc.collect()  
  print('End of sampling - bootstrap.')

def dropout(hparams):
  
  output_dir = util.create_run_dir('outputs/last_layer/', hparams)
  util.write_to_csv(os.path.join(output_dir, 'hparams.csv'), hparams)
  
  dataset_train, dataset_val = input_data(hparams)
    
  epochs = hparams['epochs']
  lr = hparams['lr']
  batch_size = hparams['batch_size']
  samples = hparams['samples']
  p_dropout = hparams['p_dropout']
  nb_last_layers = hparams['nb_last_layers']

  model = build_last_layer(p_dropout=p_dropout,
                           num_last_layers=nb_last_layers)
  
  model_path = 'saved_models/{}/{}_last_layer_{}.h5'.format(hparams['dataset'], 
                             hparams['dataset'].split('-')[0], nb_last_layers)

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

def launch_training(argv):
  
  nb_last_layers = int(argv[1])
  
  hparams = {'dataset': 'imagenet-first-1000',
             'samples': 20,
             'batch_size': 512,
             'nb_last_layers': nb_last_layers,
             }
  
  hist = initial_training(hparams)

def main(argv):

  algorithm = 'bootstrap' # argv[1]
  nb_last_layers = int(argv[1])
  lr = float(argv[2])
  
#  algorithm = 'dropout'  
#  nb_last_layers = 1

  batch_size = 512 
  dataset = 'imagenet-first-1000'
  
  hparams = {'dataset': dataset,
             'algorithm': algorithm,
             'epochs': 10, 
             'batch_size': batch_size,
             'lr': lr, # modified later on
             'p_dropout': 0.5, # modified later on
             'samples': 10, 
             'nb_last_layers': nb_last_layers,
            }
  
  if algorithm == 'sgdsgld': 
    list_lr = [0.1, 0.01, 0.001, 1e-4]
  elif algorithm == 'dropout':
    list_lr = [0.1, 0.01, 0.001, 1e-4]
    list_p_dropout = [0.5] 
  elif algorithm == 'onepoint':
    onepoint(hparams)
    return
  elif algorithm == 'bootstrap':
    bootstrap(hparams)
    return
  else:
    raise ValueError('this algorithm is not supported')
 
  i = 0
  def smartprint(i):
    print('----------------------')
    print('End of {} step'.format(i))
    print('----------------------')
  
  for lr in list_lr:
  
    hparams['lr'] = lr
        
    if algorithm == 'dropout':
      for p_dropout in list_p_dropout:
        hparams['p_dropout'] = p_dropout
        for nb_last_layers in [3]:
          hparams['nb_last_layers'] = nb_last_layers
          dropout(hparams)
          i += 1
          smartprint(i)
    if algorithm == 'sgdsgld':
      sgd_sgld(hparams)
      i += 1
      smartprint(i)
    
    
if __name__ == '__main__':
  app.run(main) 
  
  







