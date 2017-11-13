import os
import numpy as np
import urllib
import cv2

import datetime
# stitcher
from panorama import Stitcher
from utilsImage import load_image
from transform import *
import math

if __name__ == "__main__":

    input_file_list = "input_data.txt"
    output_file = "stitched.jpg"
    fp = open(input_file_list, 'r')
    _images = [each.rstrip('\r\n') for each in fp.readlines()]

    begin_time = datetime.datetime.now()


    for x in _images:
        print x

    try:
         images = []
         for _image in _images:
            images.append(load_image(_image))

         # Stitch the first two images
         stitcher = Stitcher()
         result, kps, features, deg = stitcher.stitch([images[0], images[1]], firstTime=True)

         # stitch the result image with the image with idx until stiching all images in the array
         for idx in range(2, len(images)):
                stitcher = Stitcher()
                result, kps, features, deg = stitcher.stitch([result, images[idx]],
                                                     firstTime=False, l_ori_kps=kps,
                                                     l_features=features, l_deg=deg)

         cv2.imwrite(output_file, result)

    except Exception as e:
        print(e)

    print 'Elapsed time:', datetime.datetime.now() - begin_time
