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
    # Use a threshold alpha, to convert an original gray image into a binary map
    alpha = 115
    IBinaryMap = (img < alpha) * 1
    # Alternative: Use cv2.threshold() to create a mask with value {0, 255}
    # _, IBinaryMask = cv2.threshold(img, alpha-1, 255, cv2.THRESH_BINARY_INV)

    # Smooth the binary map by a Gaussian filter
    ksize = 5
    kernel = cv2.getGaussianKernel(ksize, ksize/3)
    Gaussian_kernel = kernel @ kernel.T
    ISmoothedMap = IBinaryMap * Gaussian_kernel * 255
    # Alternative: Use the {0, 255} mask to calculate the smoothed map
    # ISmoothedMap = IBinaryMask * Gaussian_kernel

    return ISmoothedMap



def drawLargestContour(img) -> np.ndarray:
    '''
    find the largest contour and return a new image showing this contour drawn with cv2 (stroke 2)
    :param img: preprocessed image (mostly b&w)
    :return: contour image
    '''
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_img = np.zeros(img.shape)
    list(contours).sort(key=lambda c: cv2.arcLength(c, True), reverse=True)
    cv2.drawContours(contour_img, contours, 0, 255, 2)
    '''
    # Alternative: use for-loop
    # Alternative: use contourArea() to find the largest contour
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(np.array(area))
    cv2.drawContours(contour_img, contours, max_idx, 255, 2)
    '''

    return contour_img


def getFingerContourIntersections(contour_img, x) -> np.ndarray:
    '''
    Run along a column at position x, and return the 6 intersecting y-values with the finger contours.
    (For help check Palmprint_Algnment_Helper.pdf section 2b)
    :param contour_img:
    :param x: position of the image column to run along
    :return: y-values in np.ndarray in shape (6,)
    '''
    y = []
    is_image = False
    before = 255
    length_counter = 0
    for i in range(len(contour_img)):
        if contour_img[i, x] == 255 and contour_img[i, x] == before:  # white unchanged
            if not is_image:  # check board
                continue
            else:  # in case not board, then it is contour
                length_counter = length_counter + 1  # calculate the width of white
        elif contour_img[i, x] == 0 and contour_img[i, x] != before: # from white to black
            before = contour_img[i, x]
            if not is_image:  # except board
                continue
            else:
                y.append(i-(length_counter+1)//2)  # save the middle point in last white as y
        elif contour_img[i, x] == 255 and contour_img[i, x] != before:  # from black to white
            is_image = True
            length_counter = 1
            before = contour_img[i, x]
        # in case black to black, do nothing

    return np.array(y[:6])  # Q: really the first 6 ones should be chosen? what if the board is at the beginning???


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
    pass


def getCoordinateTransform(k1, k2, k3) -> np.ndarray:
    '''
    Get a transform matrix to map points from old to new coordinate system defined by k1-3
    Hint: Use cv2 for this.
    :param k1: point in (y, x) order
    :param k2: point in (y, x) order
    :param k3: point in (y, x) order
    :return: 2x3 matrix rotation around origin by angle
    '''
    pass


def palmPrintAlignment(img):
    '''
    Transform a given image like in the paper using the helper functions above when possible
    :param img: greyscale image
    :return: transformed image
    '''

    # TODO threshold and blur

    # TODO find and draw largest contour in image

    # TODO choose two suitable columns and find 6 intersections with the finger's contour

    # TODO compute middle points from these contour intersections

    # TODO extrapolate line to find k1-3

    # TODO calculate Rotation matrix from coordinate system spanned by k1-3

    # TODO rotate the image around new origin
    pass
