import os
import re
import numpy as np
# import cv2
from PIL import Image
from keras.preprocessing import image
from sklearn.preprocessing import MultiLabelBinarizer

def generate_data(ano_files,image_files):
    classes= ['person','bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
     'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
     'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
    annotations = np.empty([0,20])
    images=[]
    mlb = MultiLabelBinarizer(classes)
    for index,file_name in enumerate(ano_files):
        #get the anotations
        fp = open (ann_dir+'/'+file_name,'r')
        ano_file=fp.read() 
        start=[m.start() for m in re.finditer('<name>', ano_file)]
        end=[m.start() for m in re.finditer('</name>', ano_file)]
        annotation=[ano_file[start[i]+6:end[i]] for i in range(len(start))]
        formatted_classes = [i for i in annotation if i in classes]
        formatted_classes = [list(set(formatted_classes))]
        labels=mlb.fit_transform(formatted_classes)
        labels = np.asarray(labels).reshape(1,20)
        annotations = np.concatenate([annotations,labels],axis=0)
        #get the images
        train_image = image.load_img(img_dir+'/'+image_files[index], target_size=(227, 227))
        train_image = image.img_to_array(train_image)
        images.append(train_image)
    return images,annotations

    
if __name__ == "__main__":
    root_dir = './VOCdevkit/VOC2012/'
    img_dir = os.path.join(root_dir, 'JPEGImages')
    ann_dir = os.path.join(root_dir, 'Annotations')

    anotation_files = os.listdir(ann_dir)
    img_files=os.listdir(img_dir)
    anotation_files.sort()
    img_files.sort()
    ano_files= anotation_files[:2000]
    image_files= img_files[:2000]
    images,annotations=generate_data(ano_files,image_files)
    np.save('labels.npy',annotations)
    print(annotations)
    np.save('X_min.npy',images)
    
    ano_files= anotation_files[13700:]
    image_files= img_files[13700:]
    images,annotations=generate_data(ano_files,image_files)
    np.save('Y_test.npy',annotations)
    np.save('X_test.npy',images)