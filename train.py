from NN.Network import Network
from Data.DataSet import VOC2007
import tensorflow as tf
import numpy as np

class Train(object):
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
		self.batch_size = 64

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
		num_object = tf.placeholder(tf.int32, [None])
		learning_rate = tf.placeholder(tf.float32, [None])

		self.network = Network()
		self.network.set_batch_size(self.batch_size)

		model = self.network.model(image)
		loss = self.network.get_loss(label, model, num_object)

		train_op = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=self.momentum)

		return image, label, num_object, learning_rate, loss, train_op

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

		X, Y, O, num_data  = self.get_data_set()
		image, label, num_object, learning_rate, loss, train_op = self.setting()

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			saver = tf.train.Saver()
			saver.restore(sess, "./_model/pre_train/pretrain.ckpt")

			#train

			num_batch = int((num_data/self.batch_size) + 0.5)
			lr = 0.001
			for e in range(self.epoch):
				if(e < 10):
					lr = self.learning_rate_start + 0.001*e
				elif(e - 10 < 75):
					lr = self.learning_rate_75
				elif(e - (10 + 75) < 30):
					lr = self.learning_rate_30
				elif(e - (10 + 75 + 30) < 30):
					lr = self.learning_rate_30_final
				else:
					break

				for i in range(num_batch):
					x, y, o = self.next_batch_data(X, Y, O)
					_ = sess.run(train_op,  feed_dict={image:x, label:y, num_object:o, learning_rate:lr})

				print("epoch "+str(e)+" loss : ", sess.run(loss, feed_dict={image: x, label: y, num_object: o, learning_rate: lr}))

			saver.save(sess, "./_model/train/train.ckpt")

		return 1

	def get_data_set(self):
		data = VOC2007()
		train_set, valid_set = data.make_dataset()

		X = [i["X"] for i in train_set]
		Y = [i["Y"] for i in train_set]
		O = [i["num_object"] for i in train_set]

		return X, Y, O, len(O)

	def next_batch_data(self, X, Y, O):
		if((self.now_batch + self.batch_size) < len(O)):
			x = X[self.now_batch : self.now_batch + self.batch_size]
			y = Y[self.now_batch: self.now_batch + self.batch_size]
			o = O[self.now_batch: self.now_batch + self.batch_size]
			self.network.set_batch_size(self.now_batch)
			self.now_batch = self.now_batch + self.batch_size
		else:
			x = X[self.now_batch:]
			y = Y[self.now_batch:]
			o = O[self.now_batch:]
			self.network.set_batch_size(len(O) - self.now_batch)
			self.now_batch = 0

		return x, y, o

class PreTrain(object):
	"""PreTrain class
    
    Pre-Train the CNN

    Constructor:
		init batch_size, learning rate

    Methods:
		setting
		trainig
    """

	def __init__(self):
		print("init pre-train")
		self.batch_size = 64
		self.learning_rate = 0.001

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

		net = Network()
		model = net.model(image, pre_train=1)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=""))

		train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

		return image, train_op
	
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

		image, train_op = self.setting()

		with tf.Session() as sess:
			saver = tf.train.Saver()
			sess.run(tf.global_variables_initializer())

			#train

			saver.save(sess, "./_model/pre_train/pretrain.ckpt")


		return 1

