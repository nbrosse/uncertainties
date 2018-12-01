#%% Imports

import keras

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from keras import backend as K
from keras.models import load_model
import numpy as np

import sgld
import last_layer
import mnist

#%% Launch functions

def launch_mnist(batch_size, optimizer, p_dropout=None):
  # Load data
  (x_train, y_train), (x_test, y_test) = mnist.input_data()
  # Model
  model = mnist.model()
  # Features
  model_path = 'saved_models/mnist.h5'
  features_train = last_layer.features_extraction(model, -2, model_path, x_train)
  features_test = last_layer.features_extraction(model, -2, model_path, x_test)
  if p_dropout is None:
    submodel = last_layer.build_last_layer(model, -2, model_path)
    path_weights = last_layer.sample_last_layer(submodel, optimizer, epochs, batch_size, 
                                                features_train, _train, features_test, y_test, thinning_interval, output_dir)
  else:
    submodel = last_layer.build_last_layer(model, -2, model_path, p_dropout)
  
# parameters
batch_size = 32
num_classes = 10
index=3 # number-1 of layers of the base_CNN
on_CPU=True
if on_CPU==True:
    epochs=10



(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

from mnist_cnn import pre_processing
x_train, y_train, x_test, y_test, input_shape=pre_processing(img_rows, img_cols, on_cpu=on_CPU, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

# extract features from conv_base model
features_train=features_extraction(model, model_path, index, x_train)
print(features_train.shape)
features_test=features_extraction(model, model_path, index, x_test)
print(features_test.shape)

top_model=top_layer(128, num_classes)
predictive_distribution_matrix=last_layer_SGD_SGLD(top_model, optimizer, loss, metrics, epochs, batch_size, features_train, y_train, features_test, y_test)
