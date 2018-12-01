### Imports

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from keras import backend as K

# parameters
batch_size = 128
num_classes = 10
epochs = 3
on_CPU=True

metrics=['accuracy']
loss=keras.losses.categorical_crossentropy
optimizer=keras.optimizers.Adadelta()

# ### Data Pre-processing
# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train.shape

def pre_processing(img_rows, img_cols, on_cpu, x_train, y_train, x_test, y_test):
    if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    if on_cpu==False:
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    if on_cpu==True:
    # subset of mnist if using only CPU
        x_train=x_train[:6000]
        y_train=y_train[:6000]
        x_test=x_test[:1000]
        y_test=y_test[:1000]
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

    return x_train, y_train, x_test, y_test, input_shape

layers_names_base=['conv_base{}'.format(i) for i in range(4)]
print(layers_names_base)

def cnn_model(input_shape, layers_names_base):
    '''pre-training phase on CNN model (From F.Chollet book > 95% accuracy with 12 epochs)'''
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape, name=layers_names_base[0]))
    model.add(Conv2D(64, (3, 3), activation='relu', name=layers_names_base[1]))
    model.add(MaxPooling2D(pool_size=(2, 2), name=layers_names_base[2]))
    model.add(Dropout(0.25, name=layers_names_base[3]))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    return model

def train_model(model,loss, optimizer, metrics, epochs, batch_size, x_train, y_train, x_test, y_test):
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model_path='./output/model/mnist_cnn_epoch%03d_batchsize%03d.h5' % (epochs, batch_size)
    model.save(model_path)
    return model_path

x_train, y_train, x_test, y_test, input_shape=pre_processing(img_rows, img_cols, on_CPU, x_train, y_train, x_test, y_test)
model=cnn_model(input_shape=input_shape, layers_names_base=layers_names_base)
saved_model_path=train_model(model, loss, optimizer, metrics, epochs, batch_size, x_train, y_train, x_test, y_test)
# error in saved_model to correct
