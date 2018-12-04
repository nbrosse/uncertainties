
from __future__ import print_function

import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Input

import numpy as np

# last layer bootstrap algo implementation
def train_with_bootstrap(model, optimizer, loss, metrics, epochs, batch_size, features_train, y_train, features_test, y_test,model_path, saving_model=True):

    '''Train a model on a bootstrap of the training data and predict the matrix of predictions on the test set'''

    index_sampling = np.random.choice(range(features_train.shape[0]),features_train.shape[0], replace=True)
    features_train_sample = features_train[index_sampling]
    y_train_sample = y_train[index_sampling]

    model.fit(features_train_sample, y_train_sample,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                validation_data=(features_test, y_test))
    predictions_matrix = model.predict(features_test)

    if saving_model == True:
        top_model.save(model_path) # change to save only weights

    return predictions_matrix


def last_layer_bootstrap(model, optimizer, loss, metrics, epochs, batch_size,features_train, y_train, y_test, predictions_number):

    num_classes=len(y_test[0])
    proba_tab = np.zeros(shape=(features_test.shape[0],num_classes,predictions_number))

    for prediction in range(predictions_number):
        proba= train_with_bootstrap(model, optimizer, loss, metrics, epochs,  batch_size, features_train, y_train, features_set, y_test, model_path)
        proba_tab[:, :, index] = proba

    return proba_tab


