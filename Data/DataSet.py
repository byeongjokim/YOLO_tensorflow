from xml.etree.ElementTree import parse
import os
import cv2
import random

class VOC2007(object):
    image_width = 448
    image_height = 448

    def __init__(self):
        print("init VOC2007 dataset")
        folder_path = "./_data/VOC2007/"
        self.xml_path = folder_path + "Annotations/"
        self.image_path = folder_path + "JPEGImages/"
        self.valid_set_path = folder_path + "trainval.txt"

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
        sett = {"imageName": None, "image": None, "Object": None, "x": None, "y": None, "w": None, "h": None}

        train_set = []
        valid_set = []

        test_set = []

        images = os.listdir(self.image_path)

        for image in images:
            fileName = os.path.splitext(image)[0]
            objects = self.parsing_xml(fileName)

            for width, height, o, x, y, w, h in objects:
                ratio_width = 224/width
                ratio_height = 224/height

                sett = {"imageName": fileName, "image": cv2.resize(cv2.imread(self.image_path+image), (self.image_width,self.image_height)), "Object": o, "x": int(x*ratio_width), "y": int(y*ratio_height), "w": round(w*ratio_width + 0.5), "h": round(h*ratio_height + 0.5)}
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
            result.append([int(size.findtext("width")), int(size.findtext("height")), object.findtext("name"), int(bndbox.findtext("xmin")), int(bndbox.findtext("ymin")), int(bndbox.findtext("xmax")) - int(bndbox.findtext("xmin")), int(bndbox.findtext("ymax")) - int(bndbox.findtext("ymin"))])
        return result
