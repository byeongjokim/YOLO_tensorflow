from NN.network import Network
import tensorflow as tf

class Train(object):
    """Train
    
    d

    Constructor:
		d

    Methods:
		d
    
    """
    
	def __init__(self):
		print("init train")
		self.batch_size = 64

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
		learning_rate = tf.placeholder(tf.float32, [])
		net = Network()
		model = net.model(image)
		loss = net.loss(model)

		train_op = tf.train.MomentumOptimizer(learning_rate=learning_rate, self.momentum)

		return image, learning_rate, train_op

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

		image, learning_rate, train_op = self.setting()

		with tf.Session() as sess:
			saver = tf.train.Saver()
			saver.restore(sess, "./_model/pre_train/pretrain.ckpt")

			#train

			saver.save(sess, "./_model/train/train.ckpt")

		return 1

class PreTrain(object):
    """Train
    
    d

    Constructor:
		d

    Methods:
		d
    
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

