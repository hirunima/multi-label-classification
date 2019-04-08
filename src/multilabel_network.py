#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:48:37 2019

@author: hirunima_j
"""
# import read_data
import os
from keras import backend as K
import tensorflow as tf
import numpy as np
from keras import layers, models, optimizers
from keras import initializers
from keras.utils import to_categorical
# from keras.utils import plot_model
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras import callbacks
# np.set_printoptions(threshold=np.nan)
from keras.utils import multi_gpu_model
import keras.callbacks as callbacks
import argparse


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
        X /= (b-a)*1.0
        X *= (d-c)
        # shift to desired output range
        X += c
        return X

    def revert_preprocessing(X):
        X = X - c
        X /= (d-c)*1.0
        X *= (b-a)
        X += a
        return X

    return preprocessing, revert_preprocessing




class CustomModelCheckpoint(callbacks.Callback):

    def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        # super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.model_for_saving = model

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model_for_saving.save_weights(filepath, overwrite=True)
                        else:
                            self.model_for_saving.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model_for_saving.save_weights(filepath, overwrite=True)
                else:
                    self.model_for_saving.save(filepath, overwrite=True)

                
def get_model():
    input_image = layers.Input(shape = (227,227,3))
    x = layers.Conv2D(32, (3, 3), padding = 'same',activation='relu')(input_image)
    x = layers.Conv2D(32, (3, 3), padding = 'same',activation='relu')(x)
    x = layers.Conv2D(32, (3, 3), padding = 'same',activation='relu')(x)
    x = layers.MaxPooling2D(pool_size = (2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), padding = 'same',activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), padding = 'same',activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), padding = 'same',activation='relu')(x)
    x = layers.MaxPooling2D(pool_size = (2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding = 'same',activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding = 'same',activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding = 'same',activation='relu')(x)
    x = layers.MaxPooling2D(pool_size = (2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), padding = 'same',activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding = 'same',activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding = 'same',activation='relu')(x)
    x = layers.MaxPooling2D(pool_size = (2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512,activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512,activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    output = layers.Dense(20, name = 'dense1')(x)
    output = layers.Activation('sigmoid',name='output_layer')(output)
    

    model = models.Model(input_image,output)
  
    
    return model

def generate_data(batch_size):
    import os
    import random
    from keras.preprocessing import image
    import numpy as np
    import re
    from sklearn.preprocessing import MultiLabelBinarizer

    root_dir = 'VOCdevkit/VOC2012/'
    img_dir = os.path.join(root_dir, 'JPEGImages')
    ann_dir = os.path.join(root_dir, 'Annotations')
    label=['person','bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
     'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train','bottle', 'chair',
           'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

    classes= ['person','bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
     'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
     'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
    
    ano_files = os.listdir(ann_dir)
    image_files=os.listdir(img_dir)
    ano_files.sort()
    image_files.sort()
    ano_files= ano_files[0:13700]
    image_files= image_files[0:13700]
#     print(ano_files[0:10], image_files[0:10])
    tmp = list(zip(image_files, ano_files))
    random.shuffle(tmp)
    image_files, ano_files = zip(*tmp)
    test_split = len(image_files)
    mlb = MultiLabelBinarizer(classes)
    a = 0.0
    b = 1.0
    c = -1.0
    d = 1.0
    while True:
        for i in range(0,int(test_split/batch_size)):
            image_batch = np.empty([0,227,227,3])
            annotations = np.empty([0,20])
            for b in range(batch_size):
                img = image.load_img(img_dir+'/'+image_files[batch_size*i+b], target_size=(227, 227))
                img = image.img_to_array(img)
                img = img/255.
                ########
#                 img = img - a
#                 img /= (b-a)
#                 img *= (d-c)
#                 img += c
                ########
                img = img.reshape(1,227,227,3)
                fp = open (ann_dir+'/'+ano_files[batch_size*i+b],'r')
                ano_file=fp.read() 
                start=[m.start() for m in re.finditer('<name>', ano_file)]
                end=[m.start() for m in re.finditer('</name>', ano_file)]
                annotation=[ano_file[start[i]+6:end[i]] for i in range(len(start))]
                formatted_classes = [i for i in annotation if i in label]
                formatted_classes = [list(set(formatted_classes))]
                labels=mlb.fit_transform(formatted_classes)
                labels = np.asarray(labels).reshape(1,20)
                image_batch = np.concatenate([image_batch,img],axis=0)
                annotations = np.concatenate([annotations,labels],axis=0)
            yield image_batch,annotations
            
def generate_val(batch_size):
    import os
    import random
    from keras.preprocessing import image
    import numpy as np
    import re
    from sklearn.preprocessing import MultiLabelBinarizer

    root_dir = 'VOCdevkit/VOC2012/'
    img_dir = os.path.join(root_dir, 'JPEGImages')
    ann_dir = os.path.join(root_dir, 'Annotations')
    label=['person','bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
     'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train','bottle', 'chair',
           'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

    classes= ['person','bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
     'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
     'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
    
    ano_files = os.listdir(ann_dir)
    image_files=os.listdir(img_dir)
    ano_files.sort()
    image_files.sort()
    ano_files= ano_files[0:13700]
    image_files= image_files[0:13700]
    tmp = list(zip(image_files, ano_files))
    random.shuffle(tmp)
    image_files, ano_files = zip(*tmp)
    test_split = len(image_files)
    mlb = MultiLabelBinarizer(classes)
    a = 0.0
    b = 1.0
    c = -1.0
    d = 1.0
    
    while True:
        for i in range(0,int(test_split/batch_size)):
            image_batch = np.empty([0,227,227,3])
            annotations = np.empty([0,20])
            for b in range(batch_size):
                img = image.load_img(img_dir+'/'+image_files[batch_size*i+b], target_size=(227, 227))
                img = image.img_to_array(img)
                img = img/255.
#                 img = img - a
#                 img /= (b-a)
#                 img *= (d-c)
#                 img += c
                img = img.reshape(1,227,227,3)
                fp = open (ann_dir+'/'+ano_files[batch_size*i+b],'r')
                ano_file=fp.read() 
                start=[m.start() for m in re.finditer('<name>', ano_file)]
                end=[m.start() for m in re.finditer('</name>', ano_file)]
                annotation=[ano_file[start[i]+6:end[i]] for i in range(len(start))]
                formatted_classes = [i for i in annotation if i in label]
                formatted_classes = [list(set(formatted_classes))]
                labels=mlb.fit_transform(formatted_classes)
                labels = np.asarray(labels).reshape(1,20)
                image_batch = np.concatenate([image_batch,img],axis=0)
                annotations = np.concatenate([annotations,labels],axis=0)
            yield image_batch,annotations


def train(model,parallel_model, args):
    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights.h5', save_weights_only=True, verbose=1)
    checkpoint2 = CustomModelCheckpoint(model, args.save_dir + '/best_weights_2.h5', save_weights_only=False,verbose=1)
    parallel_model.compile(optimizers.Adam(lr = 0.0001),loss = "binary_crossentropy", metrics = ["accuracy"])
    parallel_model.fit_generator(generator=generate_data(args.batch_size), steps_per_epoch=13700/args.batch_size, epochs=args.epochs,shuffle=True,validation_data=generate_val(args.batch_size),validation_steps=3425/args.batch_size,callbacks=[log, checkpoint,checkpoint2])

    model.save_weights(args.save_dir + '/trained_model.h5')
    model.save(args.save_dir + '/trained_model2.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilabel network")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--save_dir', default='./results')
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    model = get_model()
    model.summary()
    parallel_model = multi_gpu_model(model, gpus=3)
    
    train(model=model,parallel_model=parallel_model, args=args)