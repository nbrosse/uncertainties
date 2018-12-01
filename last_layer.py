#%% Imports

from __future__ import print_function

import os
import glob
import keras

from keras.models import Model
from keras.layers import Dropout

import numpy as np

#%% Features, last layer model

def features_extraction(model, index, model_path, input_data):
  """Extract the features from the last layer.
  
  Args: 
    model: full keras model.
    index: index of the last layer, depends on the model considered.
    model_path: path to the full saved model.
    input_data: input data for the features.
  Returns:
    features: features corresponding to the input data.
  """
  output_layer = model.get_layer(index=index)
  submodel = Model(inputs=model.input, 
                   outputs=output_layer.output)
  submodel.load_weights(model_path, by_name=True)
  features = submodel.predict(input_data)
  return features

def insert_intermediate_layer(model, layer_id, new_layer):
  """Insert an intermediate layer in keras model.
  
  Args:
    model: initial keras model
    layer_id: number of the layer 
    new_layer: new keras layer to insert at position layer_id
  Returns:
    new_model: new keras model 
  """

  layers = [l for l in model.layers]

  x = layers[0].output
  for i in range(1, len(layers)):
      if i == layer_id:
          x = new_layer(x)
      x = layers[i](x)

  new_model = Model(input=layers[0].input, output=x)
  return new_model

def build_last_layer(model, index, model_path, p_dropout=None):
  """Build the last layer keras model.
  
  For Dropout, assume the submodel last layer is of the (keras) form
  =====================  
  model = Sequential()
  model.add(Dense(num_units, activation='relu'))
  ---------------------
  model.add(Dropout(p_dropout)) # to add
  ---------------------
  model.add(Dense(num_classes, activation='softmax'))
  =====================
  
  Args:
    model: full keras model.
    index: index of the last layer, depends on the model considered.
    model_path: path to the full saved model.
    p_dropout: float between 0 and 1. Fraction of the input units to drop.
  Returns:
    submodel: last layer model.
  """
  input_layer = model.get_layer(index=index)
  submodel = Model(inputs=input_layer.input, 
                   outputs=model.output)
  submodel.load_weights(model_path, by_name=True)
  if p_dropout is not None:
    submodel = insert_intermediate_layer(submodel, 1, Dropout(p_dropout))
  return submodel

def sample_last_layer(model, optimizer, epochs, batch_size, 
                      features_train, y_train, features_test, y_test,
                      thinning_interval, output_dir):
    """Train last layer model using SGD or SGLD.
    
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
      output_dir: str, directory where to write snapshots of the weights
    """
    # Create output dir
    path_weights = os.path.join(output_dir, optimizer)
    os.makedirs(path_weights)
    # Compile and train model
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    mc = keras.callbacks.ModelCheckpoint(os.path.join(path_weights, 
                                                      'weights{epoch:03d}.h5'),
                                         save_weights_only=True, 
                                         period=thinning_interval)

    model.fit(features_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(features_test, y_test),
              callbacks=[mc])
    # Sanity check
    score = model.evaluate(features_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    return path_weights

def predict_last_layer(model, features_test, num_classes, path_weights):
  """Predict probability distributions using saved weights.
  """
  weights_files = [i for i in glob.glob(os.path.join(path_weights, '*.h5'))]
  num_samples = len(weights_files)
  proba_tab = np.zeros(shape=(features_test.shape[0], 
                              num_classes, 
                              num_samples))
  for index, weights in enumerate(weights_files):
      model.load_weights(weights)
      proba = model.predict(features_test)
      proba_tab[:, :, index] = proba

  return proba_tab