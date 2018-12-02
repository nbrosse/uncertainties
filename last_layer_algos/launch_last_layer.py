#%% Imports

import os
from absl import app
import numpy as np
import keras

import last_layer_algos.last_layer as last_layer
import utils.sgld as sgld
import representation.mnist as mnist
#import representation.cifar as cifar
#import representation.cifar100 as cifar100
import utils.util as util

#%% Launch functions

def launch_mnist_sgd_sgld(epochs, batch_size, thinning_interval, lr):
  output_dir = util.create_run_dir('outputs/last_layer/sgd_sgld/mnist')
  (x_train, y_train), (x_test, y_test) = mnist.input_data()
  model = mnist.build_model()
  model_path = 'saved_models/mnist.h5'
  features_train = last_layer.features_extraction(model, model_path, x_train)
  features_test = last_layer.features_extraction(model, model_path, x_test)
  features_shape = (features_train.shape[1],)
  submodel = last_layer.build_last_layer(model_path, features_shape, 10)
  path_dic = {}
  for opt, optimizer in zip(['sgd', 'sgld'], 
                            [keras.optimizers.SGD(lr=lr), 
                             sgld.SGLD(x_train.shape[0], lr=lr)]):
    path_weights = os.path.join(output_dir, opt)
    os.makedirs(path_weights)
    path_weights = last_layer.sgd_sgld_last_layer(submodel, optimizer, epochs, 
                                                  batch_size, features_train, 
                                                  y_train, features_test, 
                                                  y_test, thinning_interval, 
                                                  path_weights)
    path_dic[opt] = path_weights
    print('End of sampling for %s' % opt)
  print('End of sampling.')
  # Create metrics dir
  path_metrics = os.path.join(output_dir, 'metrics')
  os.makedirs(path_metrics)
  # Compute the probabilities
  for opt in ['sgd', 'sgld']:
    proba_tab = last_layer.predict_sgd_sgld_last_layer(submodel, features_test,
                                                       10, path_dic[opt])
    np.save(os.path.join(path_metrics, 'p_{}.npy'.format(opt)), proba_tab)
  print('End of computing probabilities')


def launch_mnist_dropout(epochs, batch_size, p_dropout, num_samples):
  output_dir = util.create_run_dir('outputs/last_layer/dropout/mnist')
  (x_train, y_train), (x_test, y_test) = mnist.input_data()
  model = mnist.build_model()
  model_path = 'saved_models/mnist.h5'
  features_train = last_layer.features_extraction(model, model_path, x_train)
  features_test = last_layer.features_extraction(model, model_path, x_test)
  features_shape = (features_train.shape[1],)
  submodel = last_layer.build_last_layer(model_path, features_shape, 10,
                                         p_dropout=p_dropout)
  # Training
  submodel.compile(optimizer='sgd', 
                   loss='categorical_crossentropy', 
                   metrics=['accuracy'])
  submodel.fit(features_train, y_train,
               batch_size=batch_size,
               epochs=epochs,
               verbose=1,
               validation_data=(features_test, y_test))
  # Saving the model
  submodel.save(os.path.join(output_dir, 'weights.h5'))
  # Sampling
  proba_tab = np.zeros(shape=(features_test.shape[0], 10, num_samples))
  for index in np.arange(num_samples):
    proba = submodel.predict(features_test)
    proba_tab[:, :, index] = proba
  # Create metrics dir
  path_metrics = os.path.join(output_dir, 'metrics')
  os.makedirs(path_metrics)
  # Saving the probabilities
  np.save(os.path.join(path_metrics, 'p_dropout.npy'), proba_tab)
  
def main(args):
  del args # unused args
  num_samples = 10
  epochs = 10
  batch_size = 32
  p_dropout = 0.5
  thinning_interval = 1
  lr = 1.
  launch_mnist_sgd_sgld(epochs, batch_size, thinning_interval, lr)
#  launch_mnist_dropout(epochs, batch_size, p_dropout, num_samples)
  
if __name__ == '__main__':
  app.run(main)
  
