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

    def cond(self, num, num_label, label, model, loss):
        return num < num_label



    def cal_loss_obj(self, l, m):
        """dd
        
        dd
        
        Keyword Arguments:
            l (1-D tensor): [7] => [x, y, w, h, cls, cellx, celly]
            m (1-D tensor): [32] => [x, y, w, h, c, x, y, w, h, c, 20classes, j, i]

        Returns:
            dd

        Example:
            >> dd
        """
        box_l = tf.stack([l[:4], l[:4]])
        box_p = tf.stack([m[:4], m[5:9]])

        con_l = self.iou(box_l, box_p, m[-2:])
        con_p = tf.stack([m[4], m[9]])

        cls_l = tf.one_hot(tf.cast(l[4], tf.int32), 20)
        cls_p = m[10:30]
        
        box_l = tf.concat([box_l[:, :2], tf.sqrt(box_l[:, 2:])], 1)
        box_p = tf.concat([box_p[:, :2], tf.sqrt(box_p[:, 2:])], 1)

        loss_box = tf.reduce_sum(tf.pow(tf.subtract(box_l, box_p), 2))
        loss_con = tf.reduce_sum(tf.pow(tf.subtract(con_l, con_p), 2))
        loss_cls = tf.nn.softmax_cross_entropy_with_logits(labels=cls_l, logits=cls_p)

        return tf.add(tf.multiply(tf.add(loss_box, loss_con), 5), loss_cls)

    def cal_loss_nobj(self, l, m):
        """
        Keyword Arguments:
            l (1-D tensor): [7]
            m (1-D tensor): [32]
        """
        box_l = tf.stack([l[:4], l[:4]])
        box_p = tf.stack([m[:4], m[5:9]])

        con_l = self.iou(box_l, box_p, m[-2:])
        con_p = tf.stack([m[4], m[9]])

        loss_con = tf.reduce_sum(tf.pow(tf.subtract(con_l, con_p), 2))

        return 0.5 * loss_con

    def iou(self, box_l, box_p, location):
        """
        Keyword Arguments:
            box_l (2-D tensor): [4] => [min_x, min_y, max_x, max_y]
            box_p (2-D tensor): [2, 4] => [[x, y, w, h], [x, y, w, h]]
            location (1-d tensor): [2] => [j, i]
        
        Returns:
            iou (1-D tensor): [2]
        """
        
        l = tf.concat([(box_l[:, :2] + location)*64, box_l[:, 2:4]*448], 1) #=> [realx, realy, realw, realh] * 2
        p = tf.concat([(box_p[:, :2] + location)*64, box_p[:, 2:4]*448], 1) #=> [[realx, realy, realw, realh], [realx, realy, realw, realh]]

        ll = tf.stack([box_l[:, 0] - box_l[:, 2]/2, box_l[:, 1] - box_l[:, 3]/2, box_l[:, 0] + box_l[:, 2]/2, box_l[:, 1] + box_l[:, 3]/2], 1) #=> [xmin, ymin, xmax, ymax] * 2
        pp = tf.stack([box_p[:, 0] - box_p[:, 2]/2, box_p[:, 1] - box_p[:, 3]/2, box_p[:, 0] + box_p[:, 2]/2, box_p[:, 1] + box_p[:, 3]/2], 1) #=> [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax]]

        intersection = tf.stack([
                            tf.maximum(ll[:, 0], pp[:, 0]),
                            tf.maximum(ll[:, 1], pp[:, 1]),
                            tf.minimum(ll[:, 2], pp[:, 2]),
                            tf.minimum(ll[:, 3], pp[:, 3])
                        ], 1)

        area_intersection = tf.multiply(tf.subtract(intersection[:,2], intersection[:,0]), tf.subtract(intersection[:,3], intersection[:,1]))
        area_ll = tf.multiply(tf.subtract(ll[:,2], ll[:,0]), tf.subtract(ll[:,3], ll[:,1]))
        area_pp = tf.multiply(tf.subtract(pp[:,2], pp[:,0]), tf.subtract(pp[:,3], pp[:,1]))

        area_union = tf.subtract(tf.add(area_ll, area_pp), area_intersection)
        iou = tf.nn.relu(area_intersection/area_union)
        return iou

    def body(self, num, num_label, label, model, loss):
        for i in range(7):
            for j in range(7):
                m = tf.concat([model[i][j], [j,i]], 0) #=> [x, y, w, h, c, x, y, w, h, c, 20classes, j, i]
                l = tf.cond(
                                tf.equal(tf.constant(i, tf.float32), label[num][6]) & tf.equal(tf.constant(j, tf.float32), label[num][5]),
                                lambda: self.cal_loss_obj(label[num], m),
                                lambda: self.cal_loss_nobj(label[num], m)
                            )
                loss = tf.add(loss, l)

        return num+1, num_label, label, model, loss

    def get_loss(self, labels, models):
        """dd
        
        dd
        
        Keyword Arguments:
            labels (3-D tensor): [batch_size, 20, 7] # 7 => [x, y, w, h, cls, cellx, celly]
            models (4-D tensor): [batch_size, self.cell_size, self.cell_size, self.num_label + 5 * self.num_box]

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


		