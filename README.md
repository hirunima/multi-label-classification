# Multi-class Classifier

## Prerequisites
### Framework
I have used the following framework to train and test the technical assignment.
```bash
Ubuntu 16.04.5 LTS
python >= 3.6
```
Several python software packages need to be installed for the project.
```bash
opencv >= 3.4.2
numpy
Keras
tensorflow
matplotlib.pyplot
scikit-learn
```
### Data Set
Pascal VOC 2012 dataset can be downloaded via terminal.
```bash
~/$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```
To create the test data set and to get a minimize version of the dataset for the analyzer and thresholding run,
```bash
python multilabel_testing.py
```
## Neural Network Model

For the training procedure, I used Adam optimizer with a initial learning rate of 0.0001. Since one training sample contained multiple instances of several each class, labels were multihot encoded and final loss function is a summation of binary cross entropy loss of each class. when training model support multiple GPUs at once in order to reduce the training time. Training samples were resized into (227,227,3) due to resource constraints.

### Installation

For testing, 
Download the model from [here](https://uniofmora-my.sharepoint.com/:u:/g/personal/140247b_uom_lk/EfuLWiBJx7FHg39l1O2zdcQB5nP0rFoIwPRJkQTqdIlTLQ?e=2Xkskb)

Run
```bash
python multilabel_testing.py
```
If you want to train the neural network from the scratch

```bash
python multilabel_network.py --epochs 50 --batch_size 64  --save_dir ./results
```
Best model saved into the file, **~/results**
### Results

Simple CNN model abel to achieve 99.98% on validation set on pascal VOC image set after 50 epochs.

Model Summary
![model_summary.png](https://www.dropbox.com/s/2lh2ee3p0lnt0ow/model_summary.png?dl=0&raw=1)

# Analyzing the positively predicted classes

To analyzed the positively predicted classes, an analyzer with (Guided Backpropagation and LRP-PresetAFlat) was created using the iNNvestigate tool (https://github.com/albermax/innvestigate)

To identify the corresponding regions for the given prediction, analyzer ran on the neuron of the predicted classes in the classification layer.
### Installation
iNNvestigate can be installed via terminal by,
```bash
pip install innvestigate
```

Analyzer can be run via terminal
```bash
python iNNvestigate_VOC.py
```
Results are available inside the file, **~/images**
### Results
#### Class: person
Body and the shadow of the human is embossed in both reconstructions.
![0_0_0.png](https://www.dropbox.com/s/3erquhakqlb21h7/0_0_0.png?dl=0&raw=1)

#### Class: TV Monitor
Screen of the monitor is prominently embossed in both reconstructions.
![3_19.png](https://www.dropbox.com/s/zxca9cykdnm3ah2/3_19.png?dl=0&raw=1)

#### Class: Train
Here the output neuron is sensitive to shape of the train and both windows and doors of the train
![4_13.png](https://www.dropbox.com/s/9k6c0y49gqdteu4/4_13.png?dl=0&raw=1)

#### Class: Boat
Boat is identified using the main mast of the boat and the shape of the body
![5_9.png](https://www.dropbox.com/s/4h5omcr0p92bu22/5_9.png?dl=0&raw=1)

#### Class: Dog
Contour of the dogs body is accurately identified by the output neuron even though the dog and the background consists of same color.
![6_4.png](https://www.dropbox.com/s/yqd5a2ltkbpcwz5/6_4.png?dl=0&raw=1)

#### Class: Chair
Chair is identified separately by referring to the armrests.
![6_15.png](https://www.dropbox.com/s/3vj5zfugouafgv8/6_15.png?dl=0&raw=1)

#### Class: Bottle
Shape of the bottle is predominantly embossed in the reconstructed image.
![16_14.png](https://www.dropbox.com/s/kch46v1xqh6j68p/16_14.png?dl=0&raw=1)

#### Class: Dining Table
Activation of the output neuron is consistent with all the ornaments on the table. 
![16_16.png](https://www.dropbox.com/s/ru1ft3modoy9mep/16_16.png?dl=0&raw=1)

#### Class: Cow

![28_3.png](https://www.dropbox.com/s/zjudxq5rvenga1t/28_3.png?dl=0&raw=1)

### Reference
[iNNvestigate neural networks!](https://github.com/albermax/innvestigate)














