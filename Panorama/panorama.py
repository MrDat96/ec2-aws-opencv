'''This program does stitch images, from left to right, gradually'''
import numpy as np
import imutils
import cv2
import os
from os import listdir, mkdir
from os.path import isfile, join, exists
import sys
#from utilsImage import load_image
#from transform import *
import math

import datetime

'''Each time when stitch 2 image_files, we should initialize
a new of object of class Stitcher'''

def to_spherical(img, fcl):
        rows, cols = img.shape[:2]
        print ('rows, cols:', rows, cols)

        x_mid = int(cols / 2)
        y_mid = int(rows / 2)
        s = fcl
        result = np.zeros((rows, cols, 4), dtype='uint8')

        for y in range(rows):
            y_mirror = y - y_mid
            for x in range(cols):
                x_mirror = x - x_mid
                x_ = int(s * math.atan2(x_mirror, fcl))
                y_ = int(s * math.atan2(y_mirror, math.sqrt(x_mirror * x_mirror + fcl * fcl)))
                result[y_ + y_mid, x_ + x_mid] = img[y, x]

        return result
    
class Stitcher:
    '''This init function does nothing but
    enumerate all the attributes we will use
    '''

    def __init__(self):

        '''image on the left'''
        self.image_left = None
        '''original position of the keypoints of the left image'''
        self.l_ori_kps = None
        '''position of the left image's keypoints after bent out of shape
        (the degree of bending is l_deg)'''
        self.l_kps = None
        '''descriptors of the corresponding keypoints'''
        self.l_features = None
        '''degree of bending (the focal length)'''
        self.l_deg = None
        '''The indexes of the left image's keypoints in matches'''
        self.l_pts = None

        '''Same as above, except this is for the right image'''
        self.image_right = None
        self.r_ori_kps = None
        self.r_kps = None
        self.r_features = None
        self.r_pts = None

        '''Homography matrix for stitching image_files'''
        self.homo = None
        '''The matches of keypoints in 2 image_files'''
        self.matches = None
        '''Status of the matches'''
        self.status = None

        '''This attribute contains the best result so far when loop'''
        self.coolest = None

    '''Main function, do stitch 2 image_files'''
   
    
    def stitch(self, images, ratio=0.6, reprojThresh=4.0, firstTime=False,
               l_ori_kps=None, l_features=None, l_deg=None):

        def getError(self, f1, f2):

            if f1 is not None:
                # print 'f1:', f1
                self.l_kps = self.spherical_calibration(self.image_left.shape[:2], self.l_ori_kps, f1)
            self.l_pts = np.float32([self.l_kps[i] for (i, _) in self.matches])

            self.r_kps = self.spherical_calibration(self.image_right.shape[:2], self.r_ori_kps, f2)
            self.r_pts = np.float32([self.r_kps[i] for (_, i) in self.matches])
           
            (H, status) = cv2.findHomography(self.r_pts, self.l_pts, cv2.RANSAC,
                                             reprojThresh)

            absH = np.core.abs(np.copy(H))

            error = np.zeros(4)
            error[0] = 0 if absH[0, 1] < 0.02 else np.power((absH[0, 1] - 0.02) / 2000., 2)
            error[1] = 0 if absH[1, 0] < 0.005 else np.power((absH[1, 0] - 0.005) / 1000, 2)
            error[2] = 0 if absH[2, 0] < 0.00001 else np.power(absH[2, 0] - 0.00001, 2)
            error[3] = 0 if absH[2, 1] < 0.0001 else np.power((absH[2, 1] - 0.0001) / 10., 2)

            return (sum(error), H)

        '''unpack the image_files'''
        (self.image_left, self.image_right) = images

        '''if the left image had been processed, take the it's params
        if not, find the keypoint positions and descriptors as normal'''
        if not firstTime:
            self.l_ori_kps = np.copy(l_ori_kps)
            self.l_kps = np.copy(self.l_ori_kps)
            self.l_features = l_features
            self.l_deg = l_deg
        else:
            (kps, features) = self.detectAndDescribe(self.image_left)
            self.l_ori_kps = np.copy(kps)
            self.l_kps = np.copy(kps)
            self.l_features = np.copy(features)

        '''always find the the position and descriptor of keypoints in right image'''
        (kps, features) = self.detectAndDescribe(self.image_right)
        self.r_ori_kps = np.copy(kps)
        self.r_kps = np.copy(kps)
        self.r_features = np.copy(features)

        '''match keypoints of these 2 image_files'''
        matchResult = self.matchKeypoints(self.r_kps, self.l_kps,
                                          self.r_features, self.l_features, ratio, reprojThresh)

        '''if 2 image_files are not matched, just concatenate them and return'''
        if matchResult is None:
            print ('image_files are not match!')
            return np.hstack((self.image_left, self.image_right))

        '''if matched successfully, extract the needed values'''
        (self.matches, H, self.status) = matchResult

        # vis = self.drawMatches(self.image_left, self.image_right, self.l_ori_kps, self.r_ori_kps, self.matches, self.status)
        # cv2.imshow('matches_', vis)

        '''the number of bending degree to try'''
        n_f_try = 20

        '''Format of self.coolest: [error, homo matrix,
        bending degree of the left image, bending degree of the right image'''
        '''Initialize the best result as a very bad result (error = 999)'''
        self.coolest = [999, None, 0, 0]

        '''If firstTime==True, we must bend both of the image_files
        else, we only need to bend the right image'''
        if firstTime:
            '''define the values of focal length we will try'''
            f_to_try = np.linspace(min(images[0].shape[0], images[0].shape[1]),
                                   max(images[0].shape[0], images[0].shape[1]),
                                   n_f_try)
            '''for each value of fcl1, we try some values of fcl2
            that are around fcl1 value'''
            for idx1, fcl1 in enumerate(f_to_try):
                for fcl2 in f_to_try[max(0, idx1 - 3):min(idx1 + 3, n_f_try)]:
                    (now_error, homo) = getError(self, fcl1, fcl2)
                    if now_error < self.coolest[0]:
                        self.coolest = [now_error, homo, fcl1, fcl2]

        else:
            '''In this case, we only try fcl2, since fcl1 is fixed
            The value of fcl2 is expected to be somehow similar
            to fcl1 (which is also known as self.l_deg'''
            f_to_try = np.linspace(self.l_deg * 0.5, self.l_deg * 1.5, n_f_try)
            fcl1 = None
            for fcl2 in f_to_try:
                (now_error, homo) = getError(self, fcl1, fcl2)
                if now_error < self.coolest[0]:
                    self.coolest = [now_error, homo, fcl1, fcl2]

        '''Save the best result'''
        self.best_f = self.coolest[2:]
        self.homo = self.coolest[1]
        '''Keep the image_files not being skew'''
        self.homo[0, 1] = self.homo[1, 0] = self.homo[2, 0] = self.homo[2, 1] = 0

        '''bend the image by their fcl'''
        if firstTime:
            image_left = to_spherical(self.image_left, self.best_f[0])
        else:
            image_left = np.copy(self.image_left)
        image_right = to_spherical(self.image_right, self.best_f[1])

        '''Trim surplus black columns and rows'''
        image_left = self.trimSurplusCols(image_left)
        image_right = self.trimSurplusCols(image_right)

        '''Wrap right image to match left image'''
        image_right_wrapped = cv2.warpPerspective(image_right, self.homo,
                                                  (image_left.shape[1] + int(self.homo[0, 2]),
                                                   image_left.shape[0]),
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        '''Paste the 2 image_files into 1, apply some blending tricks'''
        result = self.paste_4(image_left, image_right_wrapped, moved=self.homo[0, 2])
        '''Trim, if necessary'''
        result = self.trimSurplusCols(result)

        '''Since the right image's keypoints are relocated
        when the right image is pasted to the same frame as left image,
        we should update their positions as well'''
        self.r_kps = [[self.r_kps[i][0] + self.homo[0, 2],
                       self.r_kps[i][1] + self.homo[1, 2]]
                      for i in range(len(self.r_kps))]

        '''Return result, and the params we will use when stitch more image'''
        return result, self.r_kps, self.r_features, self.best_f[1]

    '''Calibrate (bend) an image, by using ellipse equation'''

    def ellipse_calibration(self, shape, kps, deg):
        (rows, cols) = shape
        a = int(cols / 2)
        b = a * deg

        centre = cols / 2.
        diff_centre = lambda g: g - centre if g > centre else centre - g

        new_kps = []
        for i in range(len(kps)):
            y, x = kps[i]
            add_height = int(((1. - (1. * diff_centre(y) / a) ** 2) * (b ** 2)) ** 0.5)
            height_ratio = 1 + (add_height * 2. / rows)

            min_row_in_consider = b - add_height
            new_x = min_row_in_consider + int(x * height_ratio)

            new_kps.append([y, new_x])

        return new_kps

    '''Calibrate (bend) an image, to spherical view
    (kind of like a fisheye effect)'''

    def spherical_calibration(self, shape, kps, fcl):
        rows, cols = shape
        x_mid = int(cols / 2)
        y_mid = int(rows / 2)
        s = fcl

        new_kps = []
        for i in range(len(kps)):
            x, y = kps[i]

            y_mirror = y - y_mid
            x_mirror = x - x_mid

            x_ = int(s * math.atan2(x_mirror, fcl))
            y_ = int(s * math.atan2(y_mirror, math.sqrt(x_mirror * x_mirror + fcl * fcl)))

            new_kps.append([x_ + x_mid, y_ + y_mid])

        return new_kps

    '''Some blending options'''

    def paste(self, left_img, right_img, moved):
        fade_min = int(moved)
        fade_length = left_img.shape[1] - fade_min

        for j in range(left_img.shape[1]):

            if j >= fade_min:
                a_value = (1. * (left_img.shape[1] - j) / fade_length)
            else:
                a_value = 1
            for i in range(left_img.shape[0]):
                if left_img[i, j, 3] > 0:
                    right_img[i, j, :3] = left_img[i, j, :3] * a_value + (right_img[i, j, :3] * (1 - a_value)).astype(
                        int)
                    right_img[i, j, 3] = 255

        kernel = np.array([[1, 1, 1, 1, 1],
                           [1, 2, 2, 2, 1],
                           [1, 2, 5, 2, 1],
                           [1, 2, 2, 2, 1],
                           [1, 1, 1, 1, 1]]) / 37.

        right_img[:, (fade_min):(fade_min + 3), :3] = cv2.filter2D(src=right_img[:, (fade_min):(fade_min + 3), :3],
                                                                   ddepth=-1, kernel=kernel)
        # right_img[:, (fade_min - 10):(fade_min + 10), :3] = 255


        return right_img

    def paste_2(self, left_img, right_img, moved):

        centre_left = left_img.shape[1] / 2
        centre_right = (right_img.shape[1] + moved) / 2
        real_cols_right = (right_img.shape[1] - moved)
        real_cols_left = left_img.shape[1]

        rows, cols = right_img.shape[:2]

        getDiffLeff = lambda g: 1. * (g - centre_left) / real_cols_left if g > centre_left else 1. * (
            centre_left - g) / real_cols_left
        getDiffRight = lambda g: 1. * (g - centre_right) / real_cols_right if g > centre_right else 1. * (
            centre_right - g) / real_cols_right

        for i in range(rows):
            for j in range(cols):
                if j >= real_cols_left or left_img[i, j, 0] == 0:
                    pass
                elif right_img[i, j, 0] == 0:
                    right_img[i, j, :] = left_img[i, j, :]

                else:
                    prop_l = 1. - getDiffLeff(j)
                    prop_r = 1. - getDiffRight(j)
                    sum_diff = prop_l + prop_r
                    right_img[i, j, :] = (
                        left_img[i, j, :] * prop_l / sum_diff + right_img[i, j, :] * prop_r / sum_diff).astype(int)

        # right_img[:, :, 3] = 255
        return right_img

    def paste_3(self, left_img, right_img, moved):
        fade_min = int(moved)  # rightmost of left img will fade gradually
        fade_length = left_img.shape[1] - fade_min

        right_img[:, :fade_min, :] = left_img[:, :fade_min, :]

        for j in range(fade_min, left_img.shape[1]):
            for i in range(left_img.shape[0]):
                if right_img[i, j, 0] != 0 and left_img[i, j, 0] != 0:
                    right_img[i, j, :3] = (left_img[i, j, :3] + right_img[i, j, 3]) / 2
                    right_img[i, j, 3] = 255

        return right_img

    def paste_4(self, left_img, right_img, moved):
        result = np.copy(right_img)

        blend_min = int(moved)
        blend_max = left_img.shape[1]
        blend_length = float(blend_max - blend_min)

        def diff_to_mid(x):
            return (blend_max - x) / blend_length

        result[:, :blend_min + 1, :] = left_img[:, :blend_min + 1, :]

        for j in range(blend_min, blend_max):
            p = diff_to_mid(j)

            for i in range(left_img.shape[0]):
                if np.array_equal(left_img[i, j, :], [0, 0, 0, 0]):
                    result[i, j, :] = right_img[i, j, :]
                elif np.array_equal(right_img[i, j, :], [0, 0, 0, 0]):
                    result[i, j, :] = left_img[i, j, :]
                else:
                    result[i, j, :] = (left_img[i, j, :] * p + right_img[i, j, :] * (1 - p)).astype('uint8')

        return result

    '''Trim surplus columns'''

    def trimSurplusCols(self, img):

        rows, cols = img.shape[:2]

        for j in range(cols):
            meaningful = np.sum(img[:, j, 0] != 0)

            if meaningful > 0:
                left_from = j
                break

        for j in range(cols - 1, -1, -1):
            meaningful = np.sum(img[:, j, 0] != 0)

            if meaningful > 0:
                right_to = j
                break

        return img[:, left_from:right_to + 1, :]

    '''This function return keypoints'
    locations and descriptors'''

    def detectAndDescribe(self, image):

        '''detect and extract features'''
        descriptor = cv2.xfeatures2d.SIFT_create()
        kps, features = descriptor.detectAndCompute(image, None)

        '''convert the keypoints to array'''
        kps = np.float32([kp.pt for kp in kps])

        return (kps, features)

    '''From 2 sets of keypoints (of 2 image_files), find the matches'''

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            l_pts = np.float32([kpsA[i] for (_, i) in matches])
            r_pts = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(l_pts, r_pts, cv2.RANSAC,
                                             reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None

    '''Visualize the matches, use for debugging'''

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):

        matches = [(v, u) for u, v in matches]

        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 4), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis

    '''Visualize the keypoints, also use for debugging'''

    def drawKps(self, input_image, kps):
        image = np.copy(input_image)
        # print 'Drawing'
        for kp in kps:
            cv2.circle(image, (int(kp[0]), int(kp[1])), 5, (0, 255, 0))
            # cv2.imshow('zzz', image)
            # cv2.waitKey()
        return image
