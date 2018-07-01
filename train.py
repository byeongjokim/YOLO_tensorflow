from NN.Network import Network
from Data.DataSet import VOC2007, Pre_Train_Data
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
		train_X, train_Y, train_O, train_num, valid_X, valid_Y, valid_O = self.get_data_set()
		image, label, num_object, learning_rate, loss, train_op = self.setting()

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			saver = tf.train.Saver()
			saver.restore(sess, "./_model/pre_train/pretrain.ckpt")

			# train

			num_batch = int((train_num / self.batch_size) + 0.5)
			lr = 0.001
			for e in range(self.epoch):
				total_loss = 0
				if (e < 10):
					lr = self.learning_rate_start + 0.001 * e
				elif (e - 10 < 75):
					lr = self.learning_rate_75
				elif (e - (10 + 75) < 30):
					lr = self.learning_rate_30
				elif (e - (10 + 75 + 30) < 30):
					lr = self.learning_rate_30_final
				else:
					break

				for i in range(num_batch):
					x, y, o = self.next_batch_data(train_X, train_Y, train_O)
					_, loss = sess.run([train_op, loss], feed_dict={image: x, label: y, num_object: o, learning_rate: lr})

					total_loss = total_loss + loss

				print("epoch " + str(e) + " loss : " + str(total_loss))
				print("validation set loss : ", sess.run(loss, feed_dict={image: valid_X, label: valid_Y, num_object: valid_O}))

			saver.save(sess, "./_model/train/train.ckpt")

		return 1

	def get_data_set(self):
		"""d
		d
		Keyword Arguments:
			d
		Returns:
			d
		Example:
			>> d
		"""
		data = VOC2007()
		train_set, valid_set = data.make_dataset()

		X = [i["X"] for i in train_set]
		Y = [i["Y"] for i in train_set]
		O = [i["num_object"] for i in train_set]

		valid_X = [i["X"] for i in valid_set][:self.batch_size]
		valid_Y = [i["Y"] for i in valid_set][:self.batch_size]
		valid_O = [i["num_object"] for i in valid_set][:self.batch_size]

		return X, Y, O, len(O), valid_X, valid_Y, valid_O

	def next_batch_data(self, X, Y, O):
		"""d
		d
		Keyword Arguments:
			d
		Returns:
			d
		Example:
			>> d
		"""
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
		self.epoch = 200
		self.batch_size = 30
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
		label = tf.placeholder(tf.float32, [None, 20])

		net = Network()
		model = net.model(image, pre_train=1)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=label))

		is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(label, 1))
		accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

		train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

		return image, label, train_op, loss, accuracy

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
		image, label, train_op, loss, accuracy = self.setting()
		data = Pre_Train_Data()
		train_set, valid_set = data.make_dataset()

		train_X = np.array([i["X"] for i in train_set])
		train_y = np.array([i["Y"] for i in train_set])
		train_Y = np.zeros((len(train_y), 20))
		train_Y[np.arange(len(train_y)), train_y] = 1

		valid_X = np.array([i["X"] for i in valid_set])
		valid_y = np.array([i["Y"] for i in valid_set])
		valid_Y = np.zeros((len(valid_y), 20))
		valid_Y[np.arange(len(valid_y)), valid_y] = 1

		valid_X = valid_X[:100]
		valid_Y = valid_Y[:100]

		with tf.Session() as sess:
			saver = tf.train.Saver()
			sess.run(tf.global_variables_initializer())

			total_batch = int(len(train_set) / self.batch_size)

			if(total_batch == 0):
				total_batch = 1

			for e in range(self.epoch):
				total_cost = 0

				j = 0
				for i in range(total_batch):
					if (j + self.batch_size > len(train_X)):
						batch_x = train_X[j:]
						batch_y = train_Y[j:]
					else:
						batch_x = train_X[j:j + self.batch_size]
						batch_y = train_Y[j:j + self.batch_size]
						j = j + self.batch_size

					batch_x = batch_x.reshape(-1, 448, 448, 3)
					batch_y = batch_y.reshape(-1, 20)

					_, cost = sess.run([train_op, loss], feed_dict={image:batch_x, label:batch_y})

					total_cost = total_cost + cost

				print('Epoch:', '%d' % (e + 1), 'Average cost =', '{:.3f}'.format(total_cost / total_batch))

				validation_acc = sess.run(accuracy, feed_dict={image: valid_X, label: valid_Y}) * 100
				print("Validation Set Accuracy : ", validation_acc)

				if(int(validation_acc) > 80):
					break

				if (total_cost / total_batch < 0.1):
					break

			saver.save(sess, "./_model/pre_train/pretrain.ckpt")


		return 1

