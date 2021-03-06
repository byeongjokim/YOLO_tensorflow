from xml.etree.ElementTree import parse
import os
import cv2
import random
import math
import numpy as np


class VOC2007(object):
    """VOC2007 class
    Make DataSet for YOLO with VOC2007
    Constructor:
        init folder, image, xml path and make validset list
    Methods:
        make_valid_list
        make_dataset
        parsing_xml
        get_xywh_forTraining
    """

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
        """Make valid set list from trainval.txt
        To make valid set, parsing the trainval.txt and make a list.
        Save that lit in self.valid_set_list.
        This method will run at declaring the VOC2007 class.
        """
        self.valid_set_list = []
        f = open(self.valid_set_path, "r")
        while True:
            line = f.readline().rstrip()
            if not line:
                break
            self.valid_set_list.append(line)

    def make_dataset(self):
        """Make DataSet for training and validating
        parsing the folders, make dataset
        Returns:
            train_set (Dictionary list): {X: image, Y:[20, 5], O:num_object}
            valid_set (Dictionary list): {X: image, Y:[20, 5], O:num_object}
        Example:
            >> data = VOC2007()
            >> train_set, valid_set = data.make_dataset()
        """

        train_set = []
        valid_set = []

        images = os.listdir(self.image_path)

        for image in images:
            fileName = os.path.splitext(image)[0]
            objects = self.parsing_xml(fileName)

            Y = np.zeros((20, 5))
            obj = []
            num_object = len(objects)
            for width, height, o, x, y, w, h in objects:
                ratio_width = 448 / width
                ratio_height = 448 / height

                x = int(x * ratio_width)
                y = int(y * ratio_height)
                w = int(w * ratio_width)
                h = int(h * ratio_height)

                obj = [x, y, w, h, self.classes.index(o)]

                Y[self.classes.index(o)] = obj

            if (fileName in self.valid_set_list):
                valid_set.append({"X": cv2.resize(cv2.imread(self.image_path + image), (448, 448)), "Y": Y,
                                  "num_object": num_object})

            else:
                train_set.append({"X": cv2.resize(cv2.imread(self.image_path + image), (448, 448)), "Y": Y,
                                  "num_object": num_object})

        return train_set, valid_set

    def parsing_xml(self, fileName):
        """Parsing the xml
        Parsing xml file, Get data
        Keyword Arguments:
            fileName (string): xml fileName for parsing
        Returns:
            result (array): [7]
        """
        result = []

        tree = parse(self.xml_path + fileName + ".xml")
        note = tree.getroot()
        size = note.find("size")
        objects = note.findall("object")

        for object in objects:
            bndbox = object.find("bndbox")
            result.append([int(size.findtext("width")), int(size.findtext("height")),
                           object.findtext("name"),
                           self.get_center(int(bndbox.findtext("xmin")), int(bndbox.findtext("xmax"))),
                           self.get_center(int(bndbox.findtext("ymin")), int(bndbox.findtext("ymax"))),
                           int(bndbox.findtext("xmax")) - int(bndbox.findtext("xmin")),
                           int(bndbox.findtext("ymax")) - int(bndbox.findtext("ymin"))])
        return result

    def get_center(self, xmin, xmax):
        """Calculate the center of point with min and max
        int( (min + max) / 2 )
        Keyword Arguments:
            xmin (int): min of point
            xmax (int): max of point
        Returns:
            center_point (int): center of point
        """
        return int((xmin + xmax) / 2)

    def get_xywh_forTraining(self, x, y, w, h):
        """Calculate the x, y, w, h and cell
        x, y, w, h, cell for training YOLO
        Keyword Arguments:
            x (int): x of center
            y (int): y of center
            w (int): width of object
            h (int): height of object
        Returns:
            x, y (float): 0 ~ 1 in the cell
            w, h (float): 0 ~ 1 compare the object
            cell (int, int): location of the object
        """
        cell = (int(x / 64), int(y / 64))

        x = (x - cell[0] * 64) / 64
        y = (y - cell[1] * 64) / 64

        w = w / 448
        h = h / 448

        return x, y, w, h, cell

class Pre_Train_Data(object):
    def __init__(self):
        print("init pre train dataset using voc2007")
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
        """Make valid set list from trainval.txt
        To make valid set, parsing the trainval.txt and make a list.
        Save that lit in self.valid_set_list.
        This method will run at declaring the VOC2007 class.
        """
        self.valid_set_list = []
        f = open(self.valid_set_path, "r")
        while True:
            line = f.readline().rstrip()
            if not line:
                break
            self.valid_set_list.append(line)

    def make_dataset(self):
        """Make DataSet for training and validating
        parsing the folders, make dataset
        Returns:
            train_set (Dictionary list): {X: image, Y: label}
            valid_set (Dictionary list): {X: image, Y: label}
        Example:
            >> data = VOC2007()
            >> train_set, valid_set = data.make_dataset()
        """

        train_set = []
        valid_set = []

        images = os.listdir(self.image_path)[:1200]

        for image in images:
            fileName = os.path.splitext(image)[0]
            objects = self.parsing_xml(fileName)

            img = cv2.imread(self.image_path + image)

            for o, x0, y0, x1, y1 in objects:

                image_obj = img[y0:y1, x0:x1]
                label = self.classes.index(o)

                if (fileName in self.valid_set_list):
                    valid_set.append({"X": cv2.resize(image_obj, (224, 224)), "Y": label})

                else:
                    train_set.append({"X": cv2.resize(image_obj, (224, 224)), "Y": label})

        return train_set, valid_set

    def parsing_xml(self, fileName):
        """Parsing the xml
        Parsing xml file, Get data
        Keyword Arguments:
            fileName (string): xml fileName for parsing
        Returns:
            result (array): [7]
        """
        result = []

        tree = parse(self.xml_path + fileName + ".xml")
        note = tree.getroot()
        size = note.find("size")
        objects = note.findall("object")

        for object in objects:
            bndbox = object.find("bndbox")
            result.append([object.findtext("name"),
                           int(bndbox.findtext("xmin")), int(bndbox.findtext("ymin")),
                           int(bndbox.findtext("xmax")), int(bndbox.findtext("ymax")),
                           ])
        return result