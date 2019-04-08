import os
from keras import backend as K
import tensorflow as tf
import numpy as np
from keras import layers, models, optimizers
from keras import initializers
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras import callbacks
import innvestigate
import innvestigate.utils
from keras.models import load_model
import innvestigate.utils as iutils


def create_preprocessing_f(X, input_range=[0, 1]):
    """
    Generically shifts data from interval [a, b] to interval [c, d].
    Assumes that theoretical min and max values are populated.
    """

    if len(input_range) != 2:
        raise ValueError(
            "Input range must be of length 2, but was {}".format(
                len(input_range)))
    if input_range[0] >= input_range[1]:
        raise ValueError(
            "Values in input_range must be ascending. It is {}".format(
                input_range))

    a, b = X.min(), X.max()
    c, d = input_range

    def preprocessing(X):
        # shift original data to [0, b-a] (and copy)
        X = X - a
        # scale to new range gap [0, d-c]
        X /= (b-a)
        X *= (d-c)
        # shift to desired output range
        X += c
        return X

    def revert_preprocessing(X):
        X = X - c
        X /= (d-c)
        X *= (b-a)
        X += a
        return X

    return preprocessing, revert_preprocessing

def find_best_threshold(imgs_train, train_label):
    from sklearn.metrics import matthews_corrcoef
    out = model.predict(imgs_train, batch_size=10, verbose=1)

    threshold = np.arange(0.1,0.9,0.01)
    acc = []
    accuracies = []
    best_threshold = np.zeros(train_label.shape[1])
    for i in range(out.shape[1]):
        y_prob = np.array(out[:,i])
        for j in threshold:
            y_pred = np.array([1 if prob>=j else 0 for prob in y_prob])
            acc.append(matthews_corrcoef(train_label[:,i],y_pred))
        acc   = np.array(acc)

        index = np.where(acc==acc.max()) 
        accuracies.append(acc.max()) 
        best_threshold[i] = threshold[index[0][0]]
        acc = []

    np.save('best_threshold.npy', best_threshold)
    return best_threshold

if __name__ == "__main__":
    input_range = [-1,1]
    model = load_model("results4/best_weights_2.h5")
    model.summary()
    X=np.load('X_min.npy')
    labels=np.load('labels.npy')

    input_range = [-1,1]
    X /= 255.
    preprocess, revert_preprocessing = create_preprocessing_f(X, input_range)
    X2 = preprocess(X)
    th = find_best_threshold(X2, labels)
    
    testing_x=np.load('X_min.npy')
    testing_y=np.load('labels.npy')

    input_range = [-1,1]
    testing_x /= 255.
  
    prediction = model.predict(testing_x)
    predictions_final=np.empty([0,20])
    summation=0
    for i,p in enumerate(prediction):
        pred_temp=np.asarray([1. if p[j]>=th[j] else 0.  for j in range(20)]).reshape(1,20)
        predictions_final=np.concatenate([predictions_final,pred_temp],axis=0)
        if (list(pred_temp[0])==list(testing_y[i])):
            summation+=1
    print('Acc:', summation/testing_y.shape[0])