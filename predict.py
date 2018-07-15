from NN.Network import Network
import tensorflow as tf
import numpy as np
import random
import cv2

class Test(object):
    def __init__(self):
        print("init test")
        self.thresh = 0.2
        self.thresh_iou = 0.5
        self.classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]

    def setting(self):
        image = tf.placeholder(tf.float32, [None, 448, 448, 3])

        network = Network()
        model = network.model(image)

        return image, model

    def predict(self):

        img = cv2.resize(cv2.imread("./test/000009.jpg"), (448, 448))
        x = np.reshape(img, [1, 448, 448, 3])
        print(x)
        image, model = self.setting()

        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, "./_model/train/train.ckpt")

        result = sess.run(model, feed_dict={image: x})

        result = result[0]
        print(result)
        for_x = np.array([[i for i in range(7)] for j in range(7)])
        for_y = np.array([[j for i in range(7)] for j in range(7)])

        for_plus = np.stack([np.array([[i for i in range(7)] for j in range(7)]),
                             np.array([[j for i in range(7)] for j in range(7)]),
                             np.zeros([7, 7]),
                             np.zeros([7, 7])], 2)

        for_multi = np.array([[[64, 64, 448, 448] for i in range(7)] for i in range(7)])

        box1_xywh = result[:, :, :4]
        box1_xywh = np.multiply(np.add(box1_xywh, for_plus), for_multi)
        box1_xywh = np.round(box1_xywh)

        box1_confi = result[:, :, 5]
        box1_confi = np.stack([box1_confi for i in range(20)], 2)

        box2_xywh = result[:, :, 5:9]
        box2_xywh = np.multiply(np.add(box2_xywh, for_plus), for_multi)
        box2_xywh = np.round(box2_xywh)

        box2_confi = result[:, :, 10]
        box2_confi = np.stack([box2_confi for i in range(20)], 2)

        cls = result[:, :, 10:]

        box1_cls = np.multiply(cls, box1_confi)
        box2_cls = np.multiply(cls, box2_confi)
        box1_cls[box1_cls < self.thresh] = 0
        box2_cls[box2_cls < self.thresh] = 0

        box1 = np.concatenate([box1_xywh, box1_cls], 2)
        box2 = np.concatenate([box2_xywh, box2_cls], 2)

        box1 = np.reshape(box1, (49, 24))
        box2 = np.reshape(box2, (49, 24))

        boxes = np.concatenate([box1, box2])

        for i in range(4, len(boxes[0])):
            boxes = np.array(sorted(boxes, key=lambda a: a[i], reverse=True))
            new = self.nms(boxes[:, :4], boxes[:, i])
            boxes[:, i] = new

        indice_max = np.stack([np.argmax(boxes[:, 4:], axis=1), np.amax(boxes[:, 4:], axis=1)], 1)
        concat = np.concatenate([boxes[:, :4], indice_max], 1)
        result_boxes = concat[concat[:, 5] != 0]

        for box in result_boxes:
            self.make_box(box, img)

        cv2.imshow("result", img)
        cv2.waitKey(0)

    def make_box(self, bbox, img):
        x, y, w, h, label, prob = bbox

        label_name = self.classes[int(label)]

        minx = int(x - w/2)
        miny = int(y - h/2)
        maxx = int(x + w/2)
        maxy = int(y + h/2)

        cv2.rectangle(img, (minx, miny), (maxx, maxy), (255, 0, 0), 3)
        cv2.putText(img, label_name, (minx, miny), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255))

    def get_minmax_xy(self, box):
        min_x = int(box[0] - box[2] / 2)
        max_x = int(box[0] + box[2] / 2 + 0.5)
        min_y = int(box[1] - box[3] / 2)
        max_y = int(box[1] + box[3] / 2 + 0.5)

        return [min_x, min_y, max_x, max_y]

    def iou(self, box_a, box_b):
        box1 = self.get_minmax_xy(box_a)
        box2 = self.get_minmax_xy(box_b)

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        area_intersection = max((x2 - x1), 0) * max((y2 - y1), 0)
        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        area_union = area_box1 + area_box2 - area_intersection

        if(area_union > 0):
            iou = area_intersection / area_union
            return iou if iou > 0 else 0
        else:
            return 0


    def nms(self, xywh, score):
        for i in range(len(score)):
            for j in range(i + 1, len(score)):
                if (self.iou(xywh[i], xywh[j]) > 0.5):
                    score[j] = 0

        return score

