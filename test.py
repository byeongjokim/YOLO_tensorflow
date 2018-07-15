from NN.Network2 import Network
import tensorflow as tf
import numpy as np



class Test(object):
	"""Train class
    Train the YOLO Model
    Constructor:
		init batch_size, epoch, learning rate, momentum
    Methods:
		setting
		trainig
		get_data_set
		next_batch_data
    """
	def __init__(self):
		print("init train")
		self.now_batch = 0
		self.batch_size = 30
		self.epoch = 10 + 75 + 30 + 30

		self.learning_rate_start = 0.001
		self.learning_rate_75 = 0.01
		self.learning_rate_30 = 0.001
		self.learning_rate_30_final = 0.0001

		self.momentum = 0.9

	def setting(self):
		"""d
		d
		Keyword Arguments:
			d
		Returns:
			d
		Example:
			>> d
		"""
		image = tf.placeholder(tf.float32, [None, 448, 448, 3])
		label = tf.placeholder(tf.float32, [None, 20, 5])
		#num_object = tf.placeholder(tf.int32, [None])
		#learning_rate = tf.placeholder(tf.float32, [None])

		self.network = Network()
		self.network.set_batch_size(2)

		model = self.network.model(image)
		#loss = self.network.get_loss(label, model, num_object)

		#train_op = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=self.momentum)

		return image, label, model


	def training(self):
		"""d
        d
        Keyword Arguments:
        	d
        Returns:
            d
        Example:
            >> d
        """
		train_X_1 = np.zeros((448, 448, 3), dtype=float) + 1
		train_X_2 = np.zeros((448, 448, 3), dtype=float) + 2

		train_Y_1 = np.zeros((20, 5))
		train_Y_1[0][0] = 1
		train_Y_2 = np.zeros((20, 5))
		train_Y_2[0][1] = 1

		train_X = np.stack([train_X_1, train_X_2])
		train_Y = np.stack([train_Y_1, train_Y_2])
		train_numobj = np.array([1, 1])

		image, label, model = self.setting()

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			saver = tf.train.Saver()
			#saver.restore(sess, "./_model/YOLO_CNN.ckpt")

			m= sess.run(model, feed_dict={image: train_X, label: train_Y})
			print(m)

			print(saver.save(sess, "./_model/train/test.ckpt"))

		return 1

variables = tf.contrib.framework.list_variables('_model/train/test.ckpt')
for i, v in enumerate(variables):
	print("{}. name : {} \n    shape : {}".format(i, v[0], v[1]))

#a = Test()
#a.training()