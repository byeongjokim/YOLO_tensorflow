# YOLO with Tensorflow
 
 
 
## YOLO
YOLO(You Only Look Once) 
 
 
 
## Requirements
- Python 3
- TensorFlow
- NumPy
- OpenCV2
 
## Data Set
I use [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) dataset. This dataset has 20 classes "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor". I fixed this classes' order. In Data/DataSet.py, there is a source, which makes training, validation set. It parsing the xml files of VOC2007 images at first. Then load the image and resize it as (448, 448). Then re- (x, y) of center point, width, height.
 
## Model
#### CNNs
The input image's size is (448, 448, 3). It applies various filters, then get through 20 CNN layers and 4 Pooling layers and 2 FC layers. Finally its output is [7, 7, 30]. 7 is the "size of cell". 30 is "number of classes + 5(x,y,w,h,c) * number of box".
 
#### Loss
.
 
 
## Train and Test
.
