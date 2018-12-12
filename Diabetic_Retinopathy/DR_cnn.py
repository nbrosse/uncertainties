
# coding: utf-8

# ### parameters of the article
# * size: 512*512, cropping to a squared center region
# * SGD with nesterov update for batch_size=32
# * L2-regularisation to all parameters (lambda=0.001)
# * l1 reg for the last layer, same lambda 
# * Learning rate piewise constant 
# 
# * are using downsampling to balance classes
# * Possibility to load the weights of the Kaggle guy (fichier pickle): https://github.com/JeffreyDF/kaggle_diabetic_retinopathy/blob/master/dumps/2015_07_17_123003_PARAMSDUMP.pkl

# #### Results for batch_size=32, optimizer=RMSProp lr=0.001 with decay, 20 epochs
 #loss: 1.3000 - acc: 0.4842 - val_loss: 15.2551 - val_acc: 0.0536


import numpy as np
import pandas as pd
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU, Dropout
from keras import models


def pre_processing_labels(base_image_dir, is_training=True):
    if is_training:
        img_dir=os.path.join(base_image_dir,'train')
    else:
        img_dir=os.path.join(base_image_dir,'test')
        
    retina_df = pd.read_csv(os.path.join(base_image_dir, 'train/trainLabels.csv'))
    retina_df['PatientId'] = retina_df['image'].map(lambda x: x.split('_')[0])
    retina_df['path'] = retina_df['image'].map(lambda x: os.path.join(img_dir,'{}.jpeg'.format(x)))
    retina_df['exists'] = retina_df['path'].map(os.path.exists)

    print(retina_df['exists'].sum(), 'images found of', retina_df.shape[0], 'total')

    retina_df['eye'] = retina_df['image'].map(lambda x: 1 if x.split('_')[-1]=='left' else 0)
    retina_df['level_cat'] = retina_df['level'].map(lambda x: to_categorical(x, 1+retina_df['level'].max()))
    retina_df['label']=retina_df['level'].where(retina_df['level']==0,1)
    retina_df['label_cat'] = retina_df['label'].map(lambda x: to_categorical(x, 1+retina_df['label'].max()))

    df_images=retina_df[retina_df['exists']==True]
    df_images=df_images.reset_index()
    
    return df_images

def upsampling_dataset(df_images, method='categorical'):

    if method=='categorical':

        majority_class_samples=int(df_images.level.value_counts().iloc[0]*1.2)
        df_images_upsampled=df_images.groupby('level').apply(lambda x: x.sample(majority_class_samples,
                                                                                        replace=True)).reset_index(drop=True)

    elif method=='binary':

        majority_class_samples=int(df_images.label.value_counts().iloc[0]*1.2)
        df_images_upsampled=df_images.groupby('label').apply(lambda x: x.sample(majority_class_samples,
                                                                                        replace=True)).reset_index(drop=True)
    
    print('New Data Size:', df_images_upsampled.shape[0], 'Old Size:', df_images.shape[0])
    
    return df_images_upsampled


def load_data(df, resize_shape=(512,512,3),method='categorical'):
    
    shape=(len(df),)+resize_shape
    X_train=np.zeros(shape)
    if method=='categorical':
        y_train=np.zeros((len(df),len(df['level_cat'][0])))
    elif method=='binary':
        y_train=np.zeros((len(df),len(df['label_cat'][0])))
        
    for index in list(df.index):
        img_path=df.iloc[index,:]['path']
        img = Image.open(img_path)
        img=img.resize((512,512))
        X_train[index]=np.array(img)

        if method=='categorical':
            label=df.iloc[index,:]['level_cat']
        elif method=='binary':
            label=df.iloc[index,:]['label_cat']
        else:
            raise ValueError('Please enter a valid method, categorical or binary')
        y_train[index]=label
        
    print("number of samples: {}".format(X_train.shape[0]))
    return X_train, y_train


def build_model(input_shape, num_classes):

    model=models.Sequential()
    model.add(Conv2D(32, (7, 7), strides=(2,2), activation='linear',input_shape=input_shape, name='l_1'))
    model.add(LeakyReLU(alpha=0.5, name='l_2'))
    model.add(MaxPooling2D((3,3),strides=(1,1), name='l_3'))

    model.add(Conv2D(32, (3, 3), strides=(1,1), activation='linear', name='l_4'))
    model.add(LeakyReLU(alpha=0.5, name='l_5'))
    model.add(MaxPooling2D((3,3),strides=(2,2), name='l_6'))

    model.add(Conv2D(64, (3, 3), strides=(1,1), activation='linear', name='l_7'))
    model.add(LeakyReLU(alpha=0.5, name='l_8'))
    model.add(MaxPooling2D((3,3),strides=(2,2), name='l_9'))

    model.add(Conv2D(128, (3, 3), strides=(1,1), activation='linear', name='l_10'))
    model.add(LeakyReLU(alpha=0.5, name='l_11'))
    model.add(Conv2D(128, (3, 3), strides=(1,1), activation='linear', name='l_12'))
    model.add(LeakyReLU(alpha=0.5, name='l_13'))
    model.add(Conv2D(128, (3, 3), strides=(1,1), activation='linear', name='l_14'))
    model.add(LeakyReLU(alpha=0.5, name='l_15'))
    model.add(MaxPooling2D((3,3),strides=(2,2), name='l_16'))

    model.add(Conv2D(256, (3, 3), strides=(1,1), activation='linear', name='l_17'))
    model.add(LeakyReLU(alpha=0.5, name='l_18'))
    model.add(Conv2D(128, (3, 3), strides=(1,1), activation='linear', name='l_19'))
    model.add(LeakyReLU(alpha=0.5, name='l_20'))
    model.add(Conv2D(128, (3, 3), strides=(1,1), activation='linear', name='l_21'))
    model.add(LeakyReLU(alpha=0.5, name='l_22'))
    model.add(MaxPooling2D((3,3),strides=(2,2), name='l_23'))

    model.add(Flatten(name='l_24'))
    model.add(Dropout(0.5, name='l_25'))
    model.add(Dense(1024, activation='relu', name='features_layer'))
    model.add(Dense(num_classes, activation='softmax', name='ll_dense'))

    return model

