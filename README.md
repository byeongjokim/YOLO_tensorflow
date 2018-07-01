# YOLO
Tensorflow Implementation of YOLO [pdf](https://pjreddie.com/media/files/papers/yolo.pdf)
 
 
## YOLO
YOLO(You Only Look Once) 
 
 
 
## Requirements
- Python 3
- TensorFlow
- NumPy
- OpenCV2
 
## Data Set
I use [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) dataset. This dataset has 20 classes "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor". I fixed this classes' order. In Data/DataSet.py, there is a source, which makes training, validation set. It parsing the xml files of VOC2007 images at first. Then load the image and resize it as (448, 448). Then re- (x, y) of center point, width, height.
<br>
In paper, we should use ImageNet 1000 classes data when pre training. But I cannot get that data.. So I use this VOC2007 in pre training also.
 
## Usage
### Train
Donwload [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) dataset. There should be serveral folders and files (Annotations, ImageSets, JPEGImages ...) Move them to project floder.
```
~/path/to/project$ mkdir _data
~/path/to/project$ mkdir _data/VOC2007

~/VOC2007$ mv Annotations /path/to/project/_data/VOC2007
~/VOC2007$ mv JPEGImages /path/to/project/_data/VOC2007
~/VOC2007$ mv ImageSets/Layout/trainval.txt /path/to/project/_data/VOC2007
```
<br>
To Use checkpoint in Tensorflow you should make folders which the checkpoints can be saved in. There should be two folders for pre training and training.
```
~/path/to/project$ mkdir _model
~/path/to/project$ mkdir _model/pre_train
~/path/to/project$ mkdir _model/train
```
Option | Desciption
------ | ----------
`-h, --help` | show this help message and exit
`--pretrain` | option for pre-Training
`--train` | option for Training
`--test` | option for Test with Image ex) python main.py --test -i image_name
`-i, --image` | input Image for Testing
### Test
