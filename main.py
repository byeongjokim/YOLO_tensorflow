from Data.DataSet import *
from NN.Network import *
import tensorflow as tf

a = VOC2007()

c, d = a.make_dataset()

c = c[:2]
print(c)
x = [i["X"] for i in c]
y = [i["Y"] for i in c]


network = Network()
image = tf.placeholder(tf.float32, [None, 448, 448, 3])
label = tf.placeholder(tf.float32, [None, 20, 7])
model = network.model(image)
print(model)
loss = network.loss(label, model)


gpu_options = tf.GPUOptions(allow_growth=True)
session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
sess = tf.Session(config=session_config)

sess.run(tf.global_variables_initializer())

print(sess.run(loss, feed_dict={image:x, label:y}))

