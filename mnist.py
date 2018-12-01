# -*- coding: utf-8 -*-
"""Simple feedforward neural network for Mnist."""


#%% Packages

import tensorflow as tf
    
#%% Define and train the model

# https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer

def model():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(20, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  return model  

def input_data():
  (x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  y_train = tf.keras.utils.to_categorical(y_train, 10)
  y_test = tf.keras.utils.to_categorical(y_test, 10)
  return (x_train, y_train), (x_test, y_test)
  

batch_size = 32

model = model()
(x_train, y_train), (x_test, y_test) = input_data()

# Submodel
#submodel = tf.keras.models.Model(inputs=model.input,
#                                 outputs=model.get_layer(index=-2).output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20)

model_path = 'saved_models/mnist.h5'
model.save(model_path)
print('Saved trained model at %s ' % model_path)

losses = model.evaluate(x_test, y_test)

