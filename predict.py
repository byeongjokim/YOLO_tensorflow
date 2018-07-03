from NN.Network import Network
import tensorflow as tf
import numpy as np
import random
import cv2

class Test(object):
    def __init__(self):
        print("init test")


    def setting(self):
        image = tf.placeholder(tf.float32, [None, 448, 448, 3])

        self.network = Network()
        model = self.network.model(image)

        return 1

    def predict(self, image):
        cv2.resize(image, (448, 448))

        net
