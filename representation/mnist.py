# -*- coding: utf-8 -*-
"""Simple feedforward neural network for Mnist."""


#%% Packages

from absl import app

import keras

from keras.models import Sequential
from keras.layers import Dense, Flatten

    
#%% Define and train the model

def build_model():
  model = Sequential()
  model.add(Flatten(input_shape=(28, 28), name='l_1'))
  model.add(Dense(512, activation='relu', name='l_2'))
  model.add(Dense(20, activation='relu', name='features_layer'))
  model.add(Dense(10, activation='softmax', name='ll_dense'))
  return model  

def input_data():
  (x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  y_train = keras.utils.to_categorical(y_train, 10)
  y_test = keras.utils.to_categorical(y_test, 10)
  return (x_train, y_train), (x_test, y_test)
  

# Submodel
#submodel = tf.keras.models.Model(inputs=model.input,
#                                 outputs=model.get_layer(index=-2).output)


def main(argv):
  # batch_size = 32
  del argv
  model = build_model()
  (x_train, y_train), (x_test, y_test) = input_data()
  
  model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

  model.fit(x_train, y_train, epochs=20)
  
  model_path = 'saved_models/mnist.h5'
  model.save(model_path)
  print('Saved trained model at %s ' % model_path)

  losses = model.evaluate(x_test, y_test)
  return losses

if __name__ == '__main__':
  app.run(main)