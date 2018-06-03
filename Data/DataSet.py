from xml.etree.ElementTree import parse
import os
import cv2
import random
import math

class VOC2007(object):
    def __init__(self):
        print("init VOC2007 dataset")
        folder_path = "./_data/VOC2007/"
        self.xml_path = folder_path + "Annotations/"
        self.image_path = folder_path + "JPEGImages/"
        self.valid_set_path = folder_path + "trainval.txt"

        self.classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]

        self.make_valid_list()

    def make_valid_list(self):
        self.valid_set_list = []
        f = open(self.valid_set_path, "r")
        while True:
            line = f.readline().rstrip()
            if not line:
                break
            self.valid_set_list.append(line)

    def make_dataset(self):
        sett = {"imageName": None, "image": None, "objects" : [{"object": None, "cell": None, "x": None, "y": None, "w": None, "h": None}, {"object": None, "cell": None, "x": None, "y": None, "w": None, "h": None}]}

        train_set = []
        valid_set = []

        images = os.listdir(self.image_path)

        for image in images:
            fileName = os.path.splitext(image)[0]
            objects = self.parsing_xml(fileName)

            obj = []
            for width, height, o, x, y, w, h in objects:
                ratio_width = 448/width
                ratio_height = 448/height

                x = int(x * ratio_width)
                y = int(y * ratio_height)
                w = int(w * ratio_width)
                h = int(h * ratio_height)

                x, y, w, h, cell = self.get_xywh_forTraining(x, y, w, h)
                
                #bbox : x, y, w, h
                bbox = {"object" : self.classes.index(o), "cell" : cell, "x" : x, "y" : y, "w" : w, "h" : h}

                obj.append(bbox)

            sett = {"imageName": fileName, "image": cv2.resize(cv2.imread(self.image_path+image), (448, 448)), "objects" : obj}

        if(fileName in self.valid_set_list):
            valid_set.append(sett)
        else:
            train_set.append(sett)


        '''
        test = train_set[27]
        print(test["imageName"])
        print(test["Object"])
        cv2.rectangle(test["image"], (test["x"], test["y"]), (test["x"]+test["w"], test["y"]+test["h"]), (0,255,0), 3)
        cv2.imshow("asd", test["image"])
        cv2.waitKey(0)
        '''

        return train_set, valid_set


    def parsing_xml(self, fileName):
        result = []

        tree = parse(self.xml_path + fileName + ".xml")
        note = tree.getroot()
        size = note.find("size")
        objects = note.findall("object")

        for object in objects:
            bndbox = object.find("bndbox")
            result.append([int(size.findtext("width")), int(size.findtext("height")),
                           object.findtext("name"),
                           self.get_center(int(bndbox.findtext("xmin")), int(bndbox.findtext("xmax"))), self.get_center(int(bndbox.findtext("ymin")), int(bndbox.findtext("ymax"))),
                           int(bndbox.findtext("xmax")) - int(bndbox.findtext("xmin")), int(bndbox.findtext("ymax")) - int(bndbox.findtext("ymin"))])
        return result

    def get_center(self, xmin, xmax):
        return int((xmin + xmax)/2)

    def get_xywh_forTraining(self, x, y, w, h):
        cell = (int(x/64), int(y/64))

        x = (x - cell[0] * 64)/64
        y = (y - cell[1] * 64)/64

        w = w/448
        h = h/448

        return x, y, w, h, cell