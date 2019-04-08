# %load_ext autoreload
# %autoreload 2
import warnings
warnings.simplefilter('ignore')

# %matplotlib inline  

import imp
import numpy as np
import os

import keras
import keras.backend
import keras.models

import innvestigate
import innvestigate.utils as iutils
import matplotlib.pyplot as plt
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
from keras.models import load_model

# Use utility libraries to focus on relevant iNNvestigate routines.
eutils = imp.load_source("utils", "../utils.py")
mnistutils = imp.load_source("utils_mnist", "../utils_mnist.py")

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


# Scale to [0, 1] range for plotting.
def input_postprocessing(X):
    return revert_preprocessing(X2)# / 255


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

    X=np.load('X_min.npy')
    labels=np.load('labels.npy')[0:2000]
    print(X.shape, labels.shape)

    input_range = [-1,1]
    X /= 255.
    preprocess, revert_preprocessing = create_preprocessing_f(X, input_range)
    X2 = preprocess(X)

    model = load_model("results/best_weights_2.h5")
    model.summary()
    
    noise_scale = (input_range[1]-input_range[0]) * 0.1
    ri = input_range[0]  # reference input

    methods = [
        ("input",                 {},                       input_postprocessing,      "Input"),
        ("guided_backprop",       {},                       mnistutils.bk_proj,        "Guided Backprop",),
        ("lrp.sequential_preset_a_flat",{"epsilon": 1},     mnistutils.heatmap,       "LRP-PresetAFlat")
        ]

    model2 = models.Model(inputs=model.input, outputs=model.get_layer("dense1").output)

    test_images = list(zip(X2, labels, X))

    # Create analyzers.
    analyzers = []
    for method in methods:
        analyzer = innvestigate.create_analyzer(
            method[0],                     # analysis method identifier
            model2,              # model without softmax output
            neuron_selection_mode="index", # We want to select the output neuron to analyze.
            **method[1])                   # optional analysis parameters

        # Some analyzers require training.
        analyzer.fit(X2, batch_size=256, verbose=1)
        analyzers.append(analyzer)

    classes= ['person','bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
     'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
     'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
    
    th = find_best_threshold(X2[:], labels[:])
    
    prediction = model.predict(X)
    predictions_final=np.empty([0,20])
    summation=0
    correct_pred=[]
    for i,p in enumerate(prediction):
        pred_temp=np.asarray([1. if p[j]>=th[j] else 0.  for j in range(20)]).reshape(1,20)
        predictions_final=np.concatenate([predictions_final,pred_temp],axis=0)
        if (list(pred_temp[0])==list(labels[i])):
            correct_pred.append(i)
            summation+=1
    print('Acc:', summation/labels.shape[0])
    
    
    looping=True
    done=np.zeros((20,))
    for image_nr, (x, y, z) in enumerate(test_images):
        # Add batch axis.
        x = z[None, :, :, :]
        analysis = np.zeros([1, len(methods), 227, 227, 3])
        prediction = model.predict(x)
        predictions_final=[1 if prediction[0,j]>=th[j] else 0  for j in range(20)]
        if image_nr in correct_pred:
    #         print('Probabilities:',prediction[0])
    #         print("prediction:",predictions_final)
    #         print('Label:',y)
            if sum(done)==20:
                print('All the 20 classes has been analyzed!')
                break
            for ii, output_neuron in enumerate(range(0,20)):
                text = []
                if (predictions_final[ii]==1):
                    if (done[ii]==1 and sum(y)<=1):
    #                   This class is previously analyzed
                        break
                    done[ii]=1
                    # Save prediction info:
                    text.append(("%s" % classes[ii],    # ground truth label
                         "%.2f" % prediction[0][output_neuron],    # pre-softmax logits
                        ))
                    for aidx, analyzer in enumerate(analyzers):
                        # Analyze.
                        a = analyzer.analyze(x, neuron_selection=output_neuron)
                        # Apply common postprocessing, e.g., re-ordering the channels for plotting.
                        a = mnistutils.postprocess(a)
                        # Apply analysis postprocessing, e.g., creating a heatmap.
                        a = methods[aidx][2](a)
                        # Store the analysis.
                        if(aidx==0):
                            analysis[0, aidx] = x
                        else:
                            analysis[0, aidx] = a[0]
                elif (predictions_final[ii]==0):
                    continue

                # Prepare the grid as rectengular list
                grid = [[analysis[i, j] for j in range(analysis.shape[1])]
                        for i in range(analysis.shape[0])]
                print(len(grid))
                # Prepare the labels
                label, pred = zip(*text)
                row_labels_left = [('label: {}'.format(label), 'prob: {}'.format(pred))]
                row_labels_right = []
                col_labels = [''.join(method[3]) for method in methods]

                # Plot the analysis.
                save_dir="./images/"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                file_name = save_dir+str(image_nr)+'_'+str(output_neuron)
    #             plt.figure(), figsize=(10,10)
                eutils.plot_image_grid(grid, row_labels_left, row_labels_right, col_labels, file_name=file_name)
        else:
            continue
