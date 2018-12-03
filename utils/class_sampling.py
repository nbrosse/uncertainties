# imports

import numpy as np
import random

def class_sampling(x_train, y_train, num_classes, classes=None, method='random'):
    total_classes=len(y_train[0])
    classes_sampled=list(np.zeros(num_classes))

    if num_classes > total_classes:
        raise ValueError('the number of classes to sample is superior to the number of classes of the dataset')
    if num_classes==total_classes:
        print('all the classes are sampled')

    selected_index=[]
    if classes==None:
        if method=='first':
            classes_sampled=list(range(0,num_classes))
            for index in range(y_train.shape[0]):
                if np.sum(y_train[index][:num_classes])==1:
                    selected_index.append(index)
        elif method=='last':
            classes_sampled=list(range(total_classes-num_classes, total_classes))
            for index in range(y_train.shape[0]):
                if np.sum(y_train[index][-num_classes:])==1:
                    selected_index.append(index)
        elif method=='random':
            classes_sampled=list(np.random.choice(range(total_classes), num_classes, replace=False))
            for index in range(y_train.shape[0]):
                class_values=[y_train[index][cl] for cl in classes_sampled]
                if np.sum(class_values)==1:
                    selected_index.append(index)
    else:
        classes_sampled=classes
        for index in range(y_train.shape[0]):
            class_values=[y_train[index][cl] for cl in classes_sampled]
            if np.sum(class_values)==1:
                selected_index.append(index)

    x_train_restricted=x_train[selected_index]
    y_train_restricted=y_train[selected_index]
    # re_index the sampled datasets?
    oos_classes=[cl for cl in list(range(total_classes)) if not cl in classes_sampled]

    return x_train_restricted, y_train_restricted, classes_sampled, oos_classes
