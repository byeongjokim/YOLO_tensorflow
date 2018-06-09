from Data.DataSet import *
from NN.Network import *
import tensorflow as tf
import cv2
import numpy as np
"""
a = VOC2007()

c, d = a.make_dataset()

c = c[:2]
print(c)
x = [i["X"] for i in c]
y = [i["Y"] for i in c]
o = [i["num_object"] for i in c]

network = Network()
image = tf.placeholder(tf.float32, [None, 448, 448, 3])
label = tf.placeholder(tf.float32, [None, 20, 5])
num_object = tf.placeholder(tf.int32, [None])

model = network.model(image)
print(model)
#loss = network.loss(label, model, num_object)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(model, feed_dict={image:x, label:y, num_object:o}))
"""
def loss_test():
    classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]
	
    min_x = 156
    min_y = 97
    max_x = 351
    max_y = 270

    width = 500
    height = 333
    ratio_width = 448/width
    ratio_height = 448/height

    X = np.zeros((1, 448, 448, 3))
    X[0] = cv2.resize(cv2.imread("_data/VOC2007/JPEGImages/000012.jpg"), (448, 448))

    Y = np.zeros((1, 20, 5))
    Y[0][0] = [int((156+351)*ratio_width/2), int((97+270)*ratio_height/2), int((351-156)*ratio_width), int((270-97)*ratio_height), classes.index("car")]
    
    o = [1]

    image = tf.placeholder(tf.float32, [None, 448, 448, 3])
    label = tf.placeholder(tf.float32, [None, 20, 5])
    num_object = tf.placeholder(tf.int32, [None])

    network = Network()
    network.set_batch_size(X.shape[0])
    model = network.model(image)
    
    loss = network.get_loss(label, model, num_object)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    #print(sess.run(model, feed_dict={image:X, label:Y, num_object:o}))
    print(sess.run(loss, feed_dict={image:X, label:Y, num_object:o}))

def loss_test2():
    data = VOC2007()
    train_set, valid_set = data.make_dataset()

    train_set = train_set[:2]

    X = [i["X"] for i in train_set]
    Y = [i["Y"] for i in train_set]
    o = [i["num_object"] for i in train_set]

    image = tf.placeholder(tf.float32, [None, 448, 448, 3])
    label = tf.placeholder(tf.float32, [None, 20, 5])
    num_object = tf.placeholder(tf.int32, [None])

    network = Network()
    network.set_batch_size(len(X))
    model = network.model(image)
    
    loss = network.get_loss(label, model, num_object)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    #print(sess.run(model, feed_dict={image:X, label:Y, num_object:o}))
    print(sess.run(loss, feed_dict={image:X, label:Y, num_object:o}))

loss_test2()