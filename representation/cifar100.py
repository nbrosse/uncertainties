# -*- coding: utf-8 -*-
"""
Cifar 100 Keras implementation from Andrew Kruger.

References:
  https://github.com/andrewkruger/cifar100_CNN
  https://andrewkruger.github.io/projects/2017-08-05-keras-convolutional-neural-network-for-cifar-100
  
Epoch 199/200
781/781 [==============================] - 18s 24ms/step - loss: 0.2300 - acc: 0.9361 - val_loss: 2.9163 - val_acc: 0.6228
Epoch 200/200
781/781 [==============================] - 18s 24ms/step - loss: 0.2328 - acc: 0.9375 - val_loss: 2.7466 - val_acc: 0.6308
Saved trained model at saved_models/andrewkruger_cifar100.h5
10000/10000 [==============================] - 2s 207us/step
Test loss: 2.7465915702819825
Test accuracy: 0.6308
"""

#%% Imports

from __future__ import print_function
import os
from absl import app

import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import representation.cifar as cifar


#%% Model 

def build_model(x_train, num_classes):
  model = Sequential()
  
  model.add(Conv2D(128, (3, 3), padding='same',
                   input_shape=x_train.shape[1:],
                   name='l_1'))
  model.add(Activation('elu', name='l_2'))
  model.add(Conv2D(128, (3, 3), name='l_3'))
  model.add(Activation('elu', name='l_4'))
  model.add(MaxPooling2D(pool_size=(2, 2), name='l_5'))
  
  model.add(Conv2D(256, (3, 3), padding='same', name='l_6'))
  model.add(Activation('elu', name='l_7'))
  model.add(Conv2D(256, (3, 3), name='l_8'))
  model.add(Activation('elu', name='l_9'))
  model.add(MaxPooling2D(pool_size=(2, 2), name='l_10'))
  model.add(Dropout(0.25, name='l_11'))
  
  model.add(Conv2D(512, (3, 3), padding='same', name='l_12'))
  model.add(Activation('elu', name='l_13'))
  model.add(Conv2D(512, (3, 3), name='l_14'))
  model.add(Activation('elu', name='l_15'))
  model.add(MaxPooling2D(pool_size=(2, 2), name='l_16'))
  model.add(Dropout(0.25, name='l_17'))
  
  
  model.add(Flatten(name='l_18'))
  model.add(Dense(1024, activation='elu', name='features_layer'))
#  model.add(Dropout(0.5)) # because of last_layer
  model.add(Dense(num_classes, activation='softmax', name='ll_dense'))
  return model

#%% Train the model
  
def main(args):
  del args
  num_classes = 100
  save_dir = 'saved_models'
  model_name = 'andrewkruger_cifar100.h5'
  epochs = 200
  data_augmentation = True
  batch_size = 64
  
  (x_train, y_train), (x_test, y_test) = cifar.input_cifar100()
  
  # initiate RMSprop optimizer
  opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
  
  model = build_model(x_train, num_classes)

  # Let's train the model using RMSprop
  model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
  
  # Run the model
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

  model_path = os.path.join(save_dir, model_name)
  model.save(model_path)
  print('Saved trained model at %s ' % model_path)
  
  # Score trained model.
  scores = model.evaluate(x_test, y_test, verbose=1)
  print('Test loss:', scores[0])
  print('Test accuracy:', scores[1])
  
if __name__ == '__main__':
  app.run(main)  