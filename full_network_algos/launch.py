# %% Imports

import os
from absl import app
import h5py
import itertools

import numpy as np
import keras

import utils.sgld as sgld
import utils.util as util
import representation.cifar as cifar
import representation.mnist as mnist

import full_network_algos.PermaDropout_models as PD_models
from sklearn.model_selection import train_test_split

from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

# %% Algorithms

def split_in_out(method, x_train, y_train, x_val, y_val, hparams):

  n_class=hparams['n_class']

  index = util.select_classes(y_train, n_class, method=method)
  sec_train = np.dot(y_train, index).astype(bool)
  sec_val = np.dot(y_val, index).astype(bool)
  #sec_test = np.dot(y_test, index).astype(bool)

  x_train_in, x_train_out = x_train[sec_train, :], x_train[~sec_train, :]
  y_train_in = y_train[np.ix_(sec_train, index)]
  x_val_in, x_val_out = x_val[sec_val, :], x_val[~sec_val, :]
  y_val_in = y_val[np.ix_(sec_val, index)]
  #x_test_in, x_test_out = x_test[sec_test, :], x_test[~sec_test, :]
  #y_test_in = y_test[np.ix_(sec_test, index)]

  return x_train_in, y_train_in, x_val_in, x_val_out, y_val_in, index

def prepare_data_cifar(hparams):
  (x_train, y_train), (x_test, y_test) = cifar.input_data(dataset=hparams['dataset'])
  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train)
  x_train, (mean, std) = cifar.whitening(x_train)
  x_val = cifar.normalize(x_val, mean, std)

  x_train_in, y_train_in, x_val_in, x_val_out, y_val_in, index= split_in_out('first', x_train, y_train, x_val, y_val, hparams)

  # data augmentation
  datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the x_test, y_test  dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
  # (std, mean, and principal components if ZCA whitening is applied)

  hparams['dataset']=hparams['dataset']+'-first-'+'{}'.format(hparams['n_class'])
  print(hparams['dataset'])
  output_dir = util.create_run_dir('outputs/full_network/', hparams)
  util.write_to_csv(os.path.join(output_dir, 'hparams.csv'), hparams)
  np.save(os.path.join(output_dir, 'index.npy'), index)

  return x_train_in, y_train_in, x_val_in, x_val_out, y_val_in, datagen, output_dir

def prepare_data_mnist(hparams):
  (x_train, y_train), (x_test, y_test)=mnist.input_data()
  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train)
  x_train_in, y_train_in, x_val_in, x_val_out, y_val_in, index=split_in_out('first', x_train, y_train, x_val, y_val, hparams)

  hparams['dataset'] = hparams['dataset'] + '-first-' + '{}'.format(hparams['n_class'])
  print(hparams['dataset'])
  output_dir = util.create_run_dir('outputs/full_network/', hparams)
  util.write_to_csv(os.path.join(output_dir, 'hparams.csv'), hparams)
  np.save(os.path.join(output_dir, 'index.npy'), index)

  return x_train_in, y_train_in, x_val_in, x_val_out, y_val_in, output_dir

def save_probas(hparams, output_dir, x_val_in, x_val_out):
    name_in = os.path.join(output_dir, 'p_in.h5')
    file_in = h5py.File(name_in, 'a')

    shape_in = (x_val_in.shape[0], hparams['n_class'], hparams['samples'])

    proba_in = file_in.create_dataset('proba',
                                  shape_in,
                                  # dtype='f2',
                                  compression='gzip')

    if x_val_out is not None:
        name_out = os.path.join(output_dir, 'p_out.h5')
        file_out = h5py.File(name_out, 'a')
        shape_out = (x_val_out.shape[0], hparams['n_class'], hparams['samples'])
        proba_out = file_out.create_dataset('proba',
                                        shape_out,
                                        # dtype='f2',
                                        compression='gzip')
        return file_in, proba_in, file_out, proba_out
    else:
        return file_in, proba_in, None, None

def sgd_sgld(hparams):
  dataset=hparams['dataset']
  n_class = hparams['n_class']
  lr = hparams['lr']
  batch_size = hparams['batch_size']
  epochs = hparams['epochs']

  if 'mnist' in dataset:
    x_train_in, y_train_in, x_val_in, x_val_out, y_val_in, output_dir = prepare_data_mnist(hparams)
    model=mnist.build_model(n_class)
  else:
    x_train_in, y_train_in, x_val_in, x_val_out, y_val_in, datagen, output_dir = prepare_data_cifar(hparams)
    model=cifar.build_model(n_class)

  class Prediction(keras.callbacks.Callback):

      def __init__(self, params, x_val_in, x_val_out):
          super(Prediction, self).__init__()

          self.index = 0

          if x_val_out is None:
              self.out_of_dist = True
          else:
              self.out_of_dist = False

          name_in = os.path.join(params['output_dir'],
                                 'p_{}_in.h5'.format(params['optimizer']))
          self.file_in = h5py.File(name_in, 'a')

          shape_in = (x_val_in.shape[0], params['n_class'],
                      params['samples'])

          self.proba_in = self.file_in.create_dataset('proba',
                                                      shape_in,
                                                      # dtype='f2',
                                                      compression='gzip')
          self.x_val_in = x_val_in

          if not self.out_of_dist:
              name_out = os.path.join(params['output_dir'],
                                      'p_{}_out.h5'.format(params['optimizer']))
              self.file_out = h5py.File(name_out, 'a')
              shape_out = (x_val_out.shape[0], params['n_class'],
                           params['samples'])
              self.proba_out = self.file_out.create_dataset('proba',
                                                            shape_out,
                                                            # dtype='f2',
                                                            compression='gzip')
              self.x_val_out = x_val_out

      def on_epoch_end(self, epoch, logs={}):
          self.proba_in[:, :, self.index] = self.model.predict(self.x_val_in)
          if not self.out_of_dist:
              self.proba_out[:, :, self.index] = self.model.predict(self.x_val_out)
          self.index += 1

      def on_train_end(self, logs={}):
          self.file_in.close()
          if not self.out_of_dist:
              self.file_out.close()

  for opt, optimizer in zip(['sgd', 'sgld'],
                            [keras.optimizers.SGD(lr=lr),
                             sgld.SGLD(x_train_in.shape[0], lr=lr)]):

    if x_val_out is None:
      model.load_weights(hparams['weight_path'], by_name=True)

    hparams['optimizer'] = opt
    hparams['output_dir']= output_dir
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    mc = Prediction(hparams, x_val_in, x_val_out)

    if 'mnist' in dataset:
      hist=model.fit(x_train_in, y_train_in, batch_size,
                     epochs=epochs,
                     validation_data=(x_val_in, y_val_in),
                     verbose=1)
    else:
      datagen.fit(x_train_in)
      hist = model.fit_generator(datagen.flow(x_train_in, y_train_in,
                                                 batch_size=batch_size),
                      steps_per_epoch=x_train_in.shape[0] // batch_size,
                      epochs=epochs,
                      validation_data=(x_val_in, y_val_in),
                      callbacks=[mc],
                      verbose=1)
    print('End of sampling using {}'.format(opt))

  return hist


def dropout(hparams):
  dataset=hparams['dataset']
  n_class=hparams['n_class']
  epochs = hparams['epochs']
  lr = hparams['lr']
  batch_size = hparams['batch_size']
  samples = hparams['samples']
  p_dropout = hparams['p_dropout']

  if 'mnist' in dataset:
    x_train_in, y_train_in, x_val_in, x_val_out, y_val_in, output_dir = prepare_data_mnist(hparams)
    model = PD_models.mnist_model(n_class, p_dropout)
    optimizer=keras.optimizers.Adam(lr=lr)
  else:
    x_train_in, y_train_in, x_val_in, x_val_out, y_val_in, datagen, output_dir = prepare_data_cifar(hparams)
    model = PD_models.cifar_model(n_class, p_dropout)
    lr_decay = 1e-6
    lr_drop = 20
    def lr_scheduler(epoch):
      return lr * (0.5 ** (epoch // lr_drop))
    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
    optimizer = optimizers.SGD(lr=lr,
                         decay=lr_decay,
                         momentum=0.9,
                         nesterov=True)

  if x_val_out is None:
    model.load_weights(hparams['weight_path'])

  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  if 'mnist' in dataset:
    model.fit(x_train_in, y_train_in, batch_size,
              epochs=epochs,
              validation_data=(x_val_in, y_val_in),
              verbose=1)
  else:
    datagen.fit(x_train_in)
    model.fit_generator(datagen.flow(x_train_in, y_train_in,
                                          batch_size=batch_size),
                             steps_per_epoch=x_train_in.shape[0] // batch_size,
                             epochs=epochs,
                             validation_data=(x_val_in, y_val_in),
                        callbacks=[reduce_lr])

  # Sanity check
  score = model.evaluate(x_val_in, y_val_in, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
  print('End of training')

  file_in, proba_in, file_out, proba_out=save_probas(hparams, output_dir, x_val_in, x_val_out)

  for i in np.arange(samples):
    # computing probabilities
    proba_in[:, :, i] = model.predict(x_val_in)
    if x_val_out is not None:
        proba_out[:, :, i] = model.predict(x_val_out)

  file_in.close()
  if x_val_out is not None:
    file_out.close()

  print('End of sampling - dropout.')

def bootstrap(hparams):
  dataset = hparams['dataset']
  n_class = hparams['n_class']
  epochs = hparams['epochs']
  lr = hparams['lr']
  batch_size = hparams['batch_size']
  samples = hparams['samples']

  if 'mnist' in dataset:
    x_train_in, y_train_in, x_val_in, x_val_out, y_val_in, output_dir = prepare_data_mnist(hparams)
    model = mnist.build_model(n_class)
    optimizer = keras.optimizers.Adam(lr=lr)
  else:
    x_train_in, y_train_in, x_val_in, x_val_out, y_val_in, datagen, output_dir = prepare_data_cifar(hparams)
    model = cifar.build_model(n_class)
    lr_decay = 1e-6
    lr_drop = 20
    def lr_scheduler(epoch):
      return lr * (0.5 ** (epoch // lr_drop))
    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
    optimizer = optimizers.SGD(lr=lr,
                               decay=lr_decay,
                               momentum=0.9,
                               nesterov=True)

  if x_val_out is None:
    model.load_weights(hparams['weight_path'])

  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  file_in, proba_in, file_out, proba_out = save_probas(hparams, output_dir, x_val_in, x_val_out)

  for i in np.arange(samples):
    bootstrap_x_train, bootstrap_y_train= util.bootstrap(x_train_in, y_train_in)
    if 'mnist' in dataset:
      model.fit(bootstrap_x_train, bootstrap_y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_val_in, y_val_in))
    else:
      datagen.fit(bootstrap_x_train)
      model.fit_generator(datagen.flow(bootstrap_x_train, bootstrap_y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train_in.shape[0] // batch_size,
                            epochs=epochs,
                            validation_data=(x_val_in, y_val_in),
                            callbacks=[reduce_lr])


    # Sanity check
    score = model.evaluate(x_val_in, y_val_in, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('End of boostrap {}'.format(i))

    # computing probabilities
    proba_in[:, :, i] = model.predict(x_val_in)
    if x_val_out is not None:
      proba_out[:, :, i] = model.predict(x_val_out)

  file_in.close()
  if x_val_out is not None:
    file_out.close()

  print('End of sampling - bootstrap.')


def main(argv):
  del argv
  #algorithm = argv[0]

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

  hparams = {'dataset': 'cifar10',
             'weight_path': '../saved_models/mnist.h5',
             'algorithm': 'sgdsgld',
             'epochs': 1,
             'batch_size': 64,
             'lr': 0.01,
             'p_dropout': 0.5,
             'samples': 3,
             'n_class': 10
            }

  algorithm = hparams['algorithm']

  if algorithm == 'bootstrap':
    bootstrap(hparams)

  elif algorithm == 'dropout':
    dropout(hparams)

  elif algorithm == 'sgdsgld':
    sgd_sgld(hparams)
  else:
    raise ValueError('this algorithm is not supported')

  #  if algo == 'sgdsgld':
  #    # 15 sim
  #    list_dataset = ['cifar100']
  #    list_algorithms = ['sgdsgld']
  #    list_samples = [10, 100, 1000]
  #    list_batch_size = [128]
  #    list_lr = [0.001, 0.005, 0.0001, 0.00005, 0.00001]
  #    list_epochs = [100]
  #    list_p_dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
  #    list_n_class= [50,100]

  #  elif algo == 'dropout':
  #    # 45
  #    list_dataset = ['cifar100']
  #    list_algorithms = ['dropout']
  #    list_samples = [10, 100, 1000]
  #    list_batch_size = [128]
  #    list_lr = [0.001, 0.005, 0.0001, 0.00005, 0.00001]
  #    list_epochs = [100]
  #    list_p_dropout = [0.1, 0.3, 0.5]
  #    list_n_class = [50, 100]

  #  elif algo == 'bootstrap':
  #    # 10
  #    list_dataset = ['cifar100']
  #    list_algorithms = ['bootstrap']
  #    list_samples = [10, 100]
  #    list_batch_size = [128]
  #    list_lr = [0.001, 0.005, 0.0001, 0.00005, 0.00001]
  #
  #    list_epochs = [10]
  #    list_p_dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
  #    list_n_class = [50, 100]
  #  else:
  #    raise ValueError('this algorithm is not supported')
  #
  #
  #
  #  i = 0
  #  def smartprint(i):
  #    print('----------------------')
  #    print('End of {} step'.format(i))
  #    print('----------------------')
  #
  #  for dataset, algorithm, samples, batch_size, lr, n_class in \
  #    itertools.product(list_dataset, list_algorithms,
  #                      list_samples, list_batch_size, list_lr, list_n_class):
  #
  #    hparams = {'dataset': dataset,
  #               'algorithm': algorithm,
  #               'epochs': 100,
  #               'batch_size': batch_size,
  #               'lr': lr,
  #               'p_dropout': 0.5,
  #               'samples': samples,
  #               'n_class': n_class
  #              }
  #
  #    if algorithm == 'bootstrap':
  #      for epochs in list_epochs:
  #        hparams['epochs'] = epochs
  #        bootstrap(hparams)
  #        i += 1
  #        smartprint(i)
  #    elif algorithm == 'dropout':
  #      for epochs, p_dropout in itertools.product(list_epochs, list_p_dropout):
  #        hparams['epochs'] = epochs
  #        hparams['p_dropout'] = p_dropout
  #        dropout(hparams)
  #        i += 1
  #        smartprint(i)
  #    elif algorithm == 'sgdsgld':
  #      sgd_sgld(hparams)
  #      i += 1
  #      smartprint(i)
  #    else:
  #      raise ValueError('this algorithm is not supported')


if __name__ == '__main__':
  app.run(main)  # mnist, cifar10, cifar100

