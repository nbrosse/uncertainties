# imports
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from keras import backend as K
from keras.models import load_model
import numpy as np

from last_layer_algo import features_extraction, top_layer, last_layer_SGD_SGLD

# parameters
batch_size = 128
num_classes = 10
index=3 # number-1 of layers of the base_CNN
on_CPU=True
if on_CPU==True:
    epochs=10

metrics=['accuracy']
loss=keras.losses.categorical_crossentropy
optimizer=keras.optimizers.Adadelta()

# upload pre-trained model
# loading the pre_trained cnn_model
model_path='./output/model/mnist_cnn.h5'
model=load_model(model_path)

 Data Pre-processing
# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from mnist_cnn import pre_processing
x_train, y_train, x_test, y_test, input_shape=pre_processing(img_rows, img_cols, on_cpu=on_CPU, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

# extract features from conv_base model
features_train=features_extraction(model, model_path, index, x_train)
print(features_train.shape)
features_test=features_extraction(model, model_path, index, x_test)
print(features_test.shape)

top_model=top_layer(128, num_classes)
predictive_distribution_matrix=last_layer_SGD_SGLD(top_model, optimizer, loss, metrics, epochs, batch_size, features_train, y_train, features_test, y_test)
