import imutils
import cv2
import numpy as np

#from transform import *


def load_image(_img, to_be_domed=False, to_be_fisheye=False, to_be_expanded=False, to_be_diminished=False,
               to_be_diminished_2=False):
    image = cv2.imread(_img, cv2.IMREAD_UNCHANGED)
    #img = imutils.resize(image, height=400)
    img = image;
    img = np.array(img)
    img[img == 0] = 1

    if img.shape[2] == 3:
        # add one more channel to the raw image, now totally 4 channels
        img1 = np.full((img.shape[0], img.shape[1], 4), 255, dtype='uint8')
        img1[:, :, :3] = img
        img = img1

    if to_be_domed:
        img = to_dome_2(img, 0.8, 0.2)

    if to_be_fisheye:
        img = to_fish_eye(img, 0.3)

    if to_be_expanded:
        img = to_expand(img, 1.5)

    if to_be_diminished:
        img = to_diminish(img, 1.6)

    if to_be_diminished_2:
        img = to_diminish_2(img, 0.7)

    return img
