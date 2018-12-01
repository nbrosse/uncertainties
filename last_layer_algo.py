from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from keras import backend as K
from keras.models import load_model
import numpy as np

def features_extraction(model, model_path, index, input_data):
    '''feature_extraction from conv_base model
    extract the base_cnn, load_weights from pre-trained network and calculate the output features'''
    output_layer=model.get_layer(index=index)
    # avoir si obligé de recréer le modèle
    sub_model=Model(inputs=model.input, outputs=output_layer.output) # only take the conv base of the model
    print(sub_model.summary())
    sub_model.load_weights(model_path,by_name=True)
    extracted_features=sub_model.predict(input_data)
    return extracted_features


### Last_layer algo
def top_layer(n_units, num_classes):
    top_model=Sequential()
    top_model.add(Flatten())
    top_model.add(Dense(n_units, activation='relu'))
    #top_model.add(Dropout(0.5))
    top_model.add(Dense(num_classes, activation='softmax'))
    return top_model

def last_layer_SGD_SGLD(top_model, optimizer, loss, metrics, epochs, batch_size, features_train, y_train, features_test, y_test):
    '''train top_layers with weights snapshots every snapshot interval
    and return predictive distribution matrix'''

    # 1. training phase with weight snapshots
    top_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    mc = keras.callbacks.ModelCheckpoint('./output/weights/weights{epoch:08d}_%s.h5' % optimizer,
                                         save_weights_only=True, period=2)

    top_model.fit(features_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(features_test, y_test),callbacks=[mc])
    #score = model.evaluate(features_test, y_test, verbose=0)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])
    top_model_path='./output/model/mnist_top_layer_epoch%03d_optim%s.h5' % (epochs, optimizer)
    top_model.save(top_model_path)

    # 2. Compute Predictive distribution matrix on the test_set
    # get each h5 files with weights
    import os
    import glob
    weight_path='./output/weights/'
    os.chdir(weight_path)
    h5_files = [i for i in glob.glob('*.h5')]

    # calculate predictive_score_distribution
    n_snapshots=len(h5_files)
    predictive_scores_distribution=np.zeros(shape=(n_snapshots,1000,10))
    for index, weight_snapshot in enumerate(h5_files):
        top_model.load_weights(weight_snapshot)
        prediction_snapshot=top_model.predict(features_test)
        predictive_scores_distribution[index]=prediction_snapshot

    return predictive_scores_distribution


#def last_layer_bootstrap()
    #'''to complete'''
    #return

#def last_layer_Dropout()
    #'''to complete'''
    #return
