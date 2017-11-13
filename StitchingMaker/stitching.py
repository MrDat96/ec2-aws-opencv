import os
import numpy as np
import urllib
import cv2
import base64
import time
import requests
import datetime

import sys
sys.path.append('../')

# stitcher
from Panorama.panorama import Stitcher
# web_services
from Panorama.utilsImage import load_image
from Panorama.transform import *
import math

# directory
DIR_CAM01 = '../Data/Cam01/cam01_{}.jpg'
DIR_CAM02 = '../Data/Cam02/cam02_{}.jpg'
DIR_PANO = '../Data/Pano/pano_{}.jpg'

MIN_INDEX = 1
MAX_INDEX = 1

def stitching():
    for i in range(MIN_INDEX, MAX_INDEX + 1, 1):
        try:
            _images = []
            _images.append(DIR_CAM01.format(i))
            _images.append(DIR_CAM02.format(i))
            images = []
            for _image in _images:
                 images.append(load_image(_image))


            stitcher = Stitcher()
            result, kps, features, deg = stitcher.stitch([images[0], images[1]], firstTime=True)

            for idx in range(2, len(images)):
                stitcher = Stitcher()
                result, kps, features, deg = stitcher.stitch([result, images[idx]],
                                                     firstTime=False, l_ori_kps=kps,
                                                     l_features=features, l_deg=deg)


            cv2.imwrite(DIR_PANO.format(i), result)
            print 'done ', i
        except Exception as e:
            print(e)

if __name__ == "__main__":
    stitching()
