from NN.network import Network
class Train(object):
	def __init__(self):
		print("init train")

		self.batch_size = 64

		self.learning_rate_start = 0.001
		self.learning_rate_75 = 0.01
		self.learning_rate_30 = 0.001
		self.learning_rate_30_final = 0.0001

		self.momentum = 0.9


	def setting(self):
		image = tf.placeholder(tf.float32, [None, 448, 448, 3])

		net = Network()
		model = net.model(image)
		loss = net.loss(model)

		train_op = 

