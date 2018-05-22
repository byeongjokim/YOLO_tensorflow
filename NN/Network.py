import tensorflow as tf

class Network(object):
	
	def __init__(self):
		print("init Network Model")
        self.cell_size = 7
        self.num_label = 20
        self.num_box = 2

	def model(self, image):
        """Returns the network model of Yolo
        
        This model is not the fast yolo but the simple yolo.
        
        Keyword Arguments:
            image (4-D tensor): [None, 448, 448, 3]
                                This should be a RGB image with (448, 448) shape.
        
        Returns:
            model (4-D tensor): [None, self.cell_size, self.cell_size, self.num_label + 5 * self.num_box]
        
        Example:
            >> image = cv2.resize(cv2.imread("path/to/image.jpg"), (448, 448))
            >> predict_model = model(image)
        """

		image = tf.placeholder(tf.float32, [None, 448, 448, 3])

        tmp = self.conv_layer(filter_size=7, fin=3, fout=64, din=image, stride=2, name="conv_1")
        tmp = self.pool(din=tmp, size=2, stride=2, option="maxpool")

        tmp = self.conv_layer(filter_size=3, fin=64, fout=192, din=tmp, stride=1, name="conv_2")
        tmp = self.pool(din=tmp, size=2, stride=2, option="maxpool")

        tmp = self.conv_layer(filter_size=1, fin=192, fout=128, din=tmp, stride=1, name="conv_3_1")
        tmp = self.conv_layer(filter_size=3, fin=128, fout=256, din=tmp, stride=1, name="conv_3_2")
        tmp = self.conv_layer(filter_size=1, fin=256, fout=256, din=tmp, stride=1, name="conv_3_3")
        tmp = self.conv_layer(filter_size=3, fin=256, fout=512, din=tmp, stride=1, name="conv_3_4")
        tmp = self.pool(din=tmp, size=2, stride=2, option="maxpool")

        for i in range(0,4):
            tmp = self.conv_layer(filter_size=1, fin=512, fout=256, din=tmp, stride=1, name="conv_4_"+str(2*i+1))
            tmp = self.conv_layer(filter_size=3, fin=256, fout=512, din=tmp, stride=1, name="conv_4_"+str(2*i+2))
        tmp = self.conv_layer(filter_size=1, fin=512, fout=512, din=tmp, stride=1, name="conv_4_9")
        tmp = self.conv_layer(filter_size=3, fin=512, fout=1024, din=tmp, stride=1, name="conv_4_10")
        tmp = self.pool(din=tmp, size=2, stride=2, option="maxpool")

        for i in range(0,2):
            tmp = self.conv_layer(filter_size=1, fin=1024, fout=512, din=tmp, stride=1, name="conv_5_"+str(2*i+1))
            tmp = self.conv_layer(filter_size=3, fin=512, fout=1024, din=tmp, stride=1, name="conv_5_"+str(2*i+2))
        tmp = self.conv_layer(filter_size=3, fin=1024, fout=1024, din=tmp, stride=1, name="conv_5_5")
        tmp = self.conv_layer(filter_size=3, fin=1024, fout=1024, din=tmp, stride=2, name="conv_5_6")

        tmp = self.conv_layer(filter_size=3, fin=1024, fout=1024, din=tmp, stride=1, name="conv_6_1")
        tmp = self.conv_layer(filter_size=3, fin=1024, fout=1024, din=tmp, stride=1, name="conv_6_2")

        reshape = tf.reshape(tmp, [tf.shape(tmp)[0], 7 * 7 * 1024])

        #FC Layer
        with tf.device("/cpu:0"):
            W1 = tf.get_variable(name="FC_1_W", shape=[7 * 7 * 1024, 4096], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable(name="FC_1_b", shape=[4096], initializer=tf.contrib.layers.xavier_initializer())
            fc1 = tf.nn.leaky_relu(tf.matmul(reshape, W1) + b1, alpha=0.1)
            fc1 = tf.nn.dropout(fc1, keep_prob=0.5)
            
            W2 = tf.get_variable(name="FC_2_W", shape=[4096, self.cell_size * self.cell_size * (self.num_label + 5 * self.num_box)], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable(name="FC_2_b", shape=[self.cell_size * self.cell_size * (self.num_label + 5 * self.num_box)], initializer=tf.contrib.layers.xavier_initializer())
            fc2 = tf.matmul(fc1, W2) + b2

            model = tf.reshape(fc2, [tf.shape(fc2)[0], self.cell_size, self.cell_size, self.num_label + 5 * self.num_box])

        return model

    def loss(self, model):
        loss = ''
        return loss

    def conv_layer(self, filter_size, fin, fout, din, stride, name):
        """Make the convolution filter and make result using tf.nn.conv2d and relu
        
        Your weight and bias have name+_W and name+_b
        
        Keyword Arguments:
            filter_size (int): size of convolution filter
            fin (int): depth of input
            fout (int): channel of convolution filter
            din (4-D tensor): [None, height, width, depth] this is input tensor
            stride (int): size of stride
            name (string): using naming the weight and bias
        
        Returns:
            R (4-D tensor): [None, height, width, depth]
        
        Example:
            >> image = tf.placeholder(tf.float32, [None, 448, 448, 3])
            >> tmp = conv_layer(filter_size=7, fin=3, fout=64, din=image, stride=2, name="conv_1")
        """

        with tf.variable_scope(name):
            W = tf.get_variable(name=name + "_W", shape=[filter_size, filter_size, fin, fout],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name=name + "_b", shape=[fout],
                                initializer=tf.contrib.layers.xavier_initializer(0.0))
            C = tf.nn.conv2d(din, W, strides=[1, stride, stride, 1], padding='SAME')
            R = tf.nn.leaky_relu(tf.nn.bias_add(C, b), alpha=0.1)
            return R

    def pool(self, din, size, stride, option='maxpool'):
        if (option == 'maxpool'):
            pool = tf.nn.max_pool(din, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME')
        elif (option == 'avrpool'):
            pool = tf.nn.avg_pool(din, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME')
        else:
            return din
        return pool


		