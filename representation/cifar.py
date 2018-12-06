'''Train a simple deep CNN on the CIFAR small images dataset.
Official model from Keras github, initially for CIFAR10.

References:
  https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
  
To explore:
  https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py

-------------------------------------------------------------------------------
CIFAR10
-------------------------------------------------------------------------------
Epoch 99/100
1562/1562 [==============================] - 20s 13ms/step - loss: 0.5287 - acc: 0.8220 - val_loss: 0.5170 - val_acc: 0.8249
Epoch 100/100
1562/1562 [==============================] - 19s 12ms/step - loss: 0.5266 - acc: 0.8230 - val_loss: 0.5465 - val_acc: 0.8158
Saved trained model at saved_models/keras_cifar10_trained_model.h5
10000/10000 [==============================] - 1s 128us/step
Test loss: 0.5464685626983643
Test accuracy: 0.8158
-------------------------------------------------------------------------------
CIFAR100
-------------------------------------------------------------------------------
Epoch 99/100
1562/1562 [==============================] - 37s 23ms/step - loss: 2.4475 - acc: 0.3789 - val_loss: 2.3579 - val_acc: 0.3997
Epoch 100/100
1562/1562 [==============================] - 37s 24ms/step - loss: 2.4566 - acc: 0.3780 - val_loss: 2.1772 - val_acc: 0.4403
Saved trained model at /home/nbrosse/saved_models/keras_cifar100_trained_model.h5 
10000/10000 [==============================] - 1s 130us/step
Test loss: 2.1772403444290163
Test accuracy: 0.4403
'''
#%% Imports

from __future__ import print_function
import os
from absl import app

import numpy as np
import keras

from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import utils.util as util

#%% Model and data


def input_cifar10():
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  # Convert class vectors to binary class matrices.
  y_train = keras.utils.to_categorical(y_train, 10)
  y_test = keras.utils.to_categorical(y_test, 10)
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  return (x_train, y_train), (x_test, y_test)

def input_cifar100():
  (x_train, y_train), (x_test, y_test) = cifar100.load_data()
  # Convert class vectors to binary class matrices.
  y_train = keras.utils.to_categorical(y_train, 100)
  y_test = keras.utils.to_categorical(y_test, 100)
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  return (x_train, y_train), (x_test, y_test)


def build_model(x_train, num_classes):
  model = Sequential()
  model.add(Conv2D(32, (3, 3), padding='same',
                   input_shape=x_train.shape[1:],
                   name='l_1'))
  model.add(Activation('relu', name='l_2'))
  model.add(Conv2D(32, (3, 3), name='l_3'))
  model.add(Activation('relu', name='l_4'))
  model.add(MaxPooling2D(pool_size=(2, 2), name='l_5'))
  model.add(Dropout(0.25, name='l_6'))
  
  model.add(Conv2D(64, (3, 3), padding='same', name='l_7'))
  model.add(Activation('relu', name='l_8'))
  model.add(Conv2D(64, (3, 3), name='l_9'))
  model.add(Activation('relu', name='l_10'))
  model.add(MaxPooling2D(pool_size=(2, 2), name='l_11'))
  model.add(Dropout(0.25, name='l_12'))
  
  model.add(Flatten(name='l_13'))
  model.add(Dense(512, activation='relu', name='features_layer'))
#  model.add(Dropout(0.5)) # because of last layer
  model.add(Dense(num_classes, activation='softmax', name='ll_dense'))
  return model

#%% Train the model

def main(argv):
  n_class = argv[0]
  batch_size = 32
  model_name = 'keras_cifar10_trained_model.h5'
  (x_train, y_train), (x_test, y_test) = input_cifar10()
  epochs = 100
  data_augmentation = True

  sec, index = util.select_classes(y_train, n_class)
  sec_test = np.dot(y_test, index).astype(bool)
  
  path_dir = 'saved_models_ood/cifar10_sec_{}'.format(n_class)
  
  np.save(os.path.join(path_dir, 'index.npy'), index)
  x_train = x_train[sec, :]
  x_test = x_test[sec_test, :]
  y_train = y_train[np.ix_(sec, index)]
  y_test = y_test[np.ix_(sec_test, index)]
  
  model = build_model(x_train, n_class)
  
  # initiate RMSprop optimizer
  opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
  
  # Let's train the model using RMSprop
  model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
  
  steps_per_epoch = int(float(x_train.shape[0]) / batch_size)

  if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
  else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)
  
    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
  
    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

  model_path = os.path.join(path_dir, model_name)
  model.save(model_path)
  print('Saved trained model at %s ' % model_path)

  # Score trained model.
  scores = model.evaluate(x_test, y_test, verbose=1)
  print('Test loss:', scores[0])
  print('Test accuracy:', scores[1])
  
if __name__ == '__main__':
  app.run(main, argv=[5])