import tensorflow as tf

class Network(object):
    """Network class
    
    Network of Yolo Model, even pre-train Model.

    Constructor:
        init cell size(default is 7), number of label(default is 20 in VOC2007), number of box(default is 2)

    Methods:
        model
        loss
        conv_layer
        pool
    """
    def __init__(self):
        print("init Network Model")
        self.pre_num_label = 20
        self.cell_size = 7
        self.num_label = 20
        self.num_box = 2

        self.batch_size = 2

    def model(self, image, pre_train=0):
        """Returns the network model of Yolo
        
        This model is not the fast yolo but the simple yolo.
        
        Keyword Arguments:
            image (4-D tensor): [None, 448, 448, 3]
                                This should be a RGB image with (448, 448) shape.
            pre_train (int): When pre-training, 1 or 0
        
        Returns:
            pre-training:
                pre_t (pre_num_label tensor) : [pre_num_label]
            re-training:
                model (4-D tensor): [None, self.cell_size, self.cell_size, self.num_label + 5 * self.num_box]
        
        Example:
            >> image = cv2.resize(cv2.imread("path/to/image.jpg"), (448, 448))
            >> asdasdas
            >> predict_model = model(image)
        """
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
        
        
        if pre_train == 1:
            pre_reshape = tf.reshape(tmp, [tf.shape(tmp)[0], 7 * 7 * 1024])

            pre_W = tf.get_variable(name="pre_t_W", shape=[7 * 7 * 1024, self.pre_num_label], initializer=tf.contrib.layers.xavier_initializer())
            pre_b = tf.get_variable(name="pre_t_b", shape=[self.pre_num_label], initializer=tf.contrib.layers.xavier_initializer())
            pre_t = tf.matmul(pre_reshape, pre_W) + pre_b
            return pre_t
        
        #Before pre-training
        #=================================================================================================================================================================================
        #After pre-training
        
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

    def confidence_loss(self, l, m):
        j = tf.cast(l[0]/64, tf.int32)
        i = tf.cast(l[1]/64, tf.int32)
        
        return 1

    def coordinate_loss(self, l, m):
        j = tf.cast(l[0]/64, tf.int32)
        i = tf.cast(l[1]/64, tf.int32)

        xy_l = tf.mod(l[:2], 64)/64
        wh_l = tf.sqrt(l[2:4]/448)
        coord_l = tf.concat([xy_l, wh_l], 0)
        coord_l = tf.concat([coord_l, coord_l], 0)

        xywh_m_1 = m[i, j, :4]
        xywh_m_2 = m[i, j, 4:]
        coord_m = tf.concat([xywh_m_1[:2], tf.sqrt(xywh_m_1[2:]), xywh_m_2[:2], tf.sqrt(xywh_m_2[2:])], 0)

        return tf.reduce_sum(tf.pow(tf.subtract(coord_l, coord_m), 2))

    def class_loss(self, l, m):
        j = tf.cast(l[0]/64, tf.int32)
        i = tf.cast(l[1]/64, tf.int32)

        cls_l = tf.one_hot(tf.cast(l[4], tf.int32), 20)
        cls_m = tf.nn.softmax(m[i, j, :])

        return tf.reduce_sum(tf.pow(tf.subtract(cls_l, cls_m), 2))

    def cond(self):
        return 1

    def body(self, label, model):
        """dd
        
        dd
        
        Keyword Arguments:
            label (1-D tensor): [5] #=> [x, y, w, h, cls]
            model (3-D tensor): [self.cell_size, self.cell_size, self.num_label + 5 * self.num_box] #=> [7, 7, 30]

        Returns:
            dd

        Example:
            >> dd
        """

        model_xywhxywh = tf.concat([model[:,:,:4], model[:,:,5:9]], 2)
        self.confidence_loss(label[:4], model_xywhxywh)
        self.coordinate_loss(label[:4], model_xywhxywh)

        self.class_loss(label, model[:, :, 10:])


        return 1


    def loss(self, labels, models):
        """dd
        
        dd
        
        Keyword Arguments:
            labels (3-D tensor): [batch_size, 20, 5] #=> [x, y, w, h, cls]
            models (4-D tensor): [batch_size, self.cell_size, self.cell_size, self.num_label + 5 * self.num_box] #=> [batch_size, 7, 7, 30]

        Returns:
            dd

        Example:
            >> dd
        """        
        num_label = tf.constant(self.num_label)
        num = tf.constant(0)

        loss = tf.constant(0, tf.float32)

        for i in range(self.batch_size):
            model = models[i, :, :, :]
            label = labels[i, :, :]
            
            loss_result = tf.while_loop(cond=self.cond, body=self.body, loop_vars=[num, num_label, label, model, loss])
            loss = tf.add(loss, loss_result[4])

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
        """adapt Pool and make Pooling Layer
        
        You can choose MaxPool or AvrPool
        
        Keyword Arguments:
            din (4-D tensor): [None, height, width, depth] this is input tensor
            size (int): size of pool filter
            stride (int): size of stride
            option (string): MaxPool : "maxpool"
                             AvrPool : "avrpool"
        
        Returns:
            pool (4-D tensor): [None, height, width, depth]
        
        Example:
            >> image = tf.placeholder(tf.float32, [None, 448, 448, 3])
            >> tmp = conv_layer(filter_size=7, fin=3, fout=64, din=image, stride=2, name="conv_1")
            >> tmp = pool(din=tmp, size=2, stride=2, option="maxpool")
        """
        if (option == 'maxpool'):
            pool = tf.nn.max_pool(din, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME')
        elif (option == 'avrpool'):
            pool = tf.nn.avg_pool(din, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME')
        else:
            return din
        return pool


		