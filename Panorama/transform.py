import numpy as np
import cv2
import imutils

# import utils.load_image
import math


def to_dome(img, _b):
    rows, cols = img.shape[:2]
    a = cols / 2
    b = int(_b * a)

    print ('rows, cols = ', str(rows), ', ', str(cols))
    print ('a, b = ', str(a), ', ', str(b))

    dome_rows = rows + b
    dome = np.full((dome_rows, cols, 4), 255, dtype='uint8')
    diff_centre = lambda x : x - a if x > a else a - x

    for j in range (cols):
        add_height = int(((1. - (1. * diff_centre(j) / a) ** 2) * (b ** 2)) ** 0.5)
        # print dome_rows - add_height - rows, rows
        dome[(dome_rows - add_height - rows) : (dome_rows - add_height), j, :3] = img[:, j, :3]

    return dome


def to_dome_2(img, _major, _minor):
    rows, cols = img.shape[:2]
    a = cols / 2
    major = int(_major * a)
    minor = int(_minor * a)

    print ('rows, cols = ', str(rows), ', ', str(cols))
    print ('a, major, minor = ', str(a), ', ', str(major), ', ', str(minor))

    dome_rows = rows + major
    dome = np.zeros((dome_rows, cols, 4), dtype='uint8')
    dome[:, :, 3] = 255
    diff_centre = lambda x: x - a if x > a else a - x

    for j in range(cols):
        diff = (1. - (1. * diff_centre(j) / a) ** 2)
        # print 'diff = ', diff
        major_add = int((diff * (major ** 2)) ** 0.5)
        minor_add = int((diff * (minor ** 2)) ** 0.5)

        # print 'major_add, minor_add = ', major_add, minor_add

        ratio = 1 + 1. * (major_add - minor_add) / rows

        # print 'ratio: ', str(ratio)

        min_row_in_consider = dome_rows - major_add - rows
        for i in range(dome_rows - major_add - rows, dome_rows - minor_add):
            dome[i, j, :] = img[int((i - min_row_in_consider) / ratio), j, :]
    return dome


def to_fish_eye(img, _b):
    rows, cols = img.shape[:2]
    # a = int(cols / 3 * 2)
    a = cols / 2
    b = int(_b * a / 2)
    centre = int(cols/2)

    # print 'rows, cols = ' + str(rows), ', ', str(cols)
    # print 'a, b = ' + str(a), ', ', str(b)

    dome_rows = rows + b + b
    dome = np.zeros((dome_rows, cols, 4), dtype='uint8')
    dome[:, :, 3] = 255
    diff_centre = lambda x : x - centre if x > centre else centre - x

    for j in range (cols):
        # if ((1. - (1. * diff_centre(j) / a) ** 2) * (b ** 2)) < 0:
        #     print j, diff_centre(j), ((1. - (1. * diff_centre(j) / a) ** 2) * (b ** 2))
        add_height = int(((1. - (1. * diff_centre(j) / a) ** 2) * (b ** 2)) ** 0.5)
        ratio = 1 + (add_height * 2. / rows)

        min_row_in_consider = b - add_height
        for i in range(b - add_height, rows + b + add_height):
            dome[i, j, :] = img[int((i - min_row_in_consider) / ratio), j, :]

    # cv2.imshow('dome', dome)
    # cv2.waitKey(0)
    return dome


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

# if __name__ == '__main__':
#     # tmp_img = imutils.resize(cv2.imread('Panorama_lab_3_pics.png', -1), height=400)
#     # img = np.full((tmp_img.shape[0], tmp_img.shape[1], 4), 255, dtype='uint8')
#     # img[:, :, :3] = tmp_img
#     img = utils.load_image('/Users/tungphung/Documents/images8/P_20160913_100320.jpg')
#
#     # dome = to_dome(img, 1.5)
#     # cv2.imshow('Dome', dome)
#
#     dome_2 = to_dome_2(img, 0.8, 0.1)
#     cv2.imshow('Dome 2', dome_2)
#
#     cv2.waitKey(0)

def to_expand(img, deg) :
    rows, cols = img.shape[:2]
    outrows = int(deg * rows)
    base = int((outrows - rows) / 2)

    print ('base =', base)

    result = np.zeros((outrows, cols, 4), dtype='uint8')

    for j in range(cols):
        add_height = int(base * j / cols)
        ratio = 1. + 2. * base * j / cols / rows

        # print base, add_height, rows
        min_consider = base - add_height
        for i in range(min_consider, base + rows + add_height):
            result[i, j, :] = img[int((i - min_consider) / ratio), j, :]

    return result

def to_diminish(img, deg) :
    rows, cols = img.shape[:2]
    outrows = rows
    base = int((rows - (rows / deg)) / 2)

    result = np.zeros((outrows, cols, 4), dtype='uint8')

    for j in range(cols):
        subtract_height = int(base * j / cols)
        ratio = 1. - 2. * base * j / cols / rows

        # print base, add_height, rows
        min_consider = subtract_height
        for i in range(min_consider, rows - min_consider - 1):
            result[i, j, :] = img[int((i - min_consider) / ratio), j, :]

    return result

def to_diminish_2(img, deg) :

    if deg == 1:
        return img
    if deg > 1:
        print ('deg should be <= 1')
        return None

    rows, cols = img.shape[:2]

    src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
    dst_points = np.float32([[0, 0], [cols - 1, int((1. - deg) * rows)], [0, rows - 1], [cols - 1, int(rows * deg)]])
    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_output = cv2.warpPerspective(img, projective_matrix, (cols, rows), borderValue=0)

    return img_output

# def to_diminish_3(img, deg) :
#
#     if deg == 1:
#         return img
#     if deg > 1:
#         print 'deg should be <= 1'
#         return None
#
#     rows, cols = img.shape[:2]
#
#     src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
#     dst_points = np.float32([[0, 0], [int(cols * deg), int((1. - deg) * rows)], [0, rows - 1], [int(cols * deg), int(rows * deg)]])
#     projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
#     img_output = cv2.warpPerspective(img, projective_matrix, (int(cols * deg) + 1, rows))
#
#     # return img_output

def make_blur(img):
    matrix = np.array([ [1,1,1,1,1],
                        [1,2,2,2,1],
                        [1,2,0,2,1],
                        [1,2,2,2,1],
                        [1,1,1,1,1]]) / 32.0
    return cv2.filter2D(img, -1, matrix)

def make_sharpen(img):
    matrix = np.array([ [-1,-1,-1,-1,-1],
                        [-1,2,2,2,-1],
                        [-1,2,8,2,-1],
                        [-1,2,2,2,-1],
                        [-1,-1,-1,-1,-1]]) / 8.0
    return cv2.filter2D(img, -1, matrix)

def fill_rec(img, final_rows):
    rows, cols = img.shape[:2]

    to_col = cols

    output_img = np.full((final_rows, cols, 4), 255, dtype='uint8')

    for j in range(cols):
        from_row = rows
        for i in range(0, rows):
            if img[i, j, 0] != 0:
                from_row = i
                break
        to_row = 0
        for i in range(rows - 1, -1, -1):
            if img[i, j, 0] != 0:
                to_row = i
                break
        if from_row > to_row:
            to_col = j - 1
            break

        # print from_row, to_row

        height = to_row - from_row + 1
        deg = 1. * height / final_rows

        for i in range(0, final_rows):
            output_img[i, j, :3] = img[int(i * deg) + from_row, j, :3]

    output_img = output_img[:, :to_col, :]

    return output_img


if __name__ == '__main__':
    img = imutils.resize(cv2.imread('/Users/tungphung/Documents/images8/P_20160913_100320.jpg', -1), height=400)

    if img.shape[2] == 3:
        img1 = np.full((img.shape[0], img.shape[1], 4), 255, dtype='uint8')
        img1[:, :, :3] = img
        img = img1

    img = to_diminish_2(img, .7)
    cv2.imshow('', img)
    cv2.waitKey(0)
