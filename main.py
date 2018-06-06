from Data.DataSet import *
from NN.Network2 import *
import tensorflow as tf

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

