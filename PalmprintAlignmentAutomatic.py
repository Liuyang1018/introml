'''
Created on 05.10.2016
Modified on 23.12.2020

@author: Daniel
@modified by Charly, Christian, Max (23.12.2020)
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt


# do not import more modules!


def drawCircle(img, x, y):
    '''
    Draw a circle at circle of radius 5px at (x, y) stroke 2 px
    This helps you to visually check your methods.
    :param img: a 2d nd-array
    :param y:
    :param x:
    :return: img with circle at desired position
    '''
    cv2.circle(img, (x, y), 5, 255, 2)
    return img


def binarizeAndSmooth(img) -> np.ndarray:
    '''
    First Binarize using threshold of 115, then smooth with gauss kernel (5, 5)
    :param img: greyscale image in range [0, 255]
    :return: preprocessed image
    '''
    a, b = np.shape(img)
    binarized = np.zeros((a, b))
    #binarize
    for i in range(a):
        for j in range(b):
            if img[i, j] > 115:
                binarized[i, j] = 255
            else:
                binarized[i, j] = 0

    preprocessed = cv2.GaussianBlur(binarized, (5, 5), 0)
    return preprocessed

def drawLargestContour(img) -> np.ndarray:
    '''
    find the largest contour and return a new image showing this contour drawn with cv2 (stroke 2)
    :param img: preprocessed image (mostly b&w)
    :return: contour image
    '''
    img = img.astype(np.uint8)
    newimg = np.zeros(np.shape(img))
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(newimg, contours, -1, 255, 2)

    return newimg


def getFingerContourIntersections(contour_img, x) -> np.ndarray:
    '''
        Run along a column at position x, and return the 6 intersecting y-values with the finger contours.
        (For help check Palmprint_Algnment_Helper.pdf section 2b)
        :param contour_img:
        :param x: position of the image column to run along
        :return: y-values in np.ndarray in shape (6,)
        '''
    flag1 = 0
    flag2 = 0
    inter = np.zeros((8,2))
    value = np.zeros((6,))
    for i in range(5, np.shape(contour_img)[0] - 1 ):
        if contour_img[i, x] == 255 and contour_img[i + 1, x] == 255 and contour_img[i - 1, x] == 0:
            inter[flag1][0] = i
            flag1 += 1
        if contour_img[i, x] == 255 and contour_img[i + 1, x] == 0 and contour_img[i - 1, x] == 255:
            inter[flag2][1] = i
            flag2 += 1

    for i in range(6):
         sum = inter[i][0] + inter[i][1]
         value[i] = sum / 2
    value = np.round(value)
    return value

def findKPoints(img, y1, x1, y2, x2) -> tuple:
    '''
    given two points and the contour image, find the intersection point k
    :param img: binarized contour image (255 == contour)
    :param y1: y-coordinate of point
    :param x1: x-coordinate of point
    :param y2: y-coordinate of point
    :param x2: x-coordinate of point
    :return: intersection point k as a tuple (ky, kx)
    '''
    ky, kx = 0, 0
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    # slope = y1 -y2 / (x1 - x2)
    # t = y1 - slope * x1
    a, b = np.shape(img)
    for j in range(1, b - 1):
        for i in range(1, a - 1):
            if kx != 0 or ky != 0:
                break
            if A * j + B * i + C == 0:
                if img[i, j] == 255:
                    ky = i
                    kx = j

    print(ky, kx)
    return (ky, kx)


def getCoordinateTransform(k1, k2, k3) -> np.ndarray:
    '''
    Get a transform matrix to map points from old to new coordinate system defined by k1-3
    Hint: Use cv2 for this.
    :param k1: point in (y, x) order
    :param k2: point in (y, x) order
    :param k3: point in (y, x) order
    :return: 2x3 matrix rotation around origin by angle
    '''
    if k1[1] - k3[1] == 0:
         slope2 = 0
         centerX = k1[1]
         centerY = k2[0]
    else:
         slope1 = (k1[0] - k3[0]) / (k1[1] - k3[1])
         slope2 = - 1 / slope1
         b1 = k1[0] - slope1 * k1[1]
         b2 = k2[0] - slope2 * k2[1]
         centerX = (b2 - b1) / (slope1 - slope2)
         centerY = centerX * slope1 + b1

    r = np.arctan(slope2)
    angle = np.degrees(r)
    M = cv2.getRotationMatrix2D((centerY,centerX), angle = angle, scale=1.0)
    return M

def palmPrintAlignment(img):
    '''
    Transform a given image like in the paper using the helper functions above when possible
    :param img: greyscale image
    :return: transformed image
    '''

    # TODO threshold and blur
    binerized = binarizeAndSmooth(img)

    # TODO find and draw largest contour in image
    contourimg = drawLargestContour(binerized)

    # TODO choose two suitable columns and find 6 intersections with the finger's contour
    v1 = getFingerContourIntersections(contourimg, 22)
    v2 = getFingerContourIntersections(contourimg, 30)
    print(v1, v2)

    # TODO compute middle points from these contour intersections
    mid20 = np.zeros((3,))
    mid30 = np.zeros((3,))
    flag = 0
    for i in range(0, 6, 2):
        mid20[flag] = (v1[i + 1] - v1[i]) / 2 + v1[i]
        mid30[flag] = (v2[i + 1] - v2[i]) / 2 + v2[i]
        flag += 1
    mid20 = np.around(mid20)
    mid30 = np.around(mid30)
    #contourimg = drawCircle(contourimg, 20,int(mid20[0]))
    #contourimg = drawCircle(contourimg, 30,int(mid30[0]))
    print('mid20:',mid20)
    print('mid30:',mid30)

    # TODO extrapolate line to find k1-3
    k1 = findKPoints(contourimg, mid20[0], 22, mid30[0], 30)
    k2 = findKPoints(contourimg, mid20[1], 22, mid30[1], 30)
    k3 = findKPoints(contourimg, mid20[2], 22, mid30[2], 30)
    contourimg = drawCircle(contourimg,k1[1], k1[0])
    contourimg = drawCircle(contourimg, k2[1], k2[0])
    contourimg = drawCircle(contourimg, k3[1], k3[0])
    #plt.imshow(contourimg, cmap='gray')
    #plt.show()

    # TODO calculate Rotation matrix from coordinate system spanned by k1-3
    m = getCoordinateTransform(k1, k2, k3)

    # TODO rotate the image around new origin
    a, b = np.shape(img)
    rotated = cv2.warpAffine(img, m, (b, a))
    return rotated
