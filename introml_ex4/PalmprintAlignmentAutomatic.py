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
    IBinaryMap = (img > alpha) * np.float64(255)

    # Alternative: Use cv2.threshold() to create a mask with value {0, 255}
    # _, IBinaryMask = cv2.threshold(img, alpha-1, 255, cv2.THRESH_BINARY_INV)

    # Smooth the binary map by a Gaussian filter
    preprocessed = cv2.GaussianBlur(IBinaryMap, (5, 5), 0)

    return preprocessed



def drawLargestContour(img) -> np.ndarray:
    '''
    find the largest contour and return a new image showing this contour drawn with cv2 (stroke 2)
    :param img: preprocessed image (mostly b&w)
    :return: contour image
    '''
    plt.imshow(img, 'gray')
    plt.show()
    img = img.astype(np.uint8)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_img = np.zeros(img.shape)
    max_contour = max(contours, key=len)
    cv2.drawContours(contour_img, max_contour, -1, 255, 2)

    '''
    # Alternative: use for-loop
    # Alternative: use contourArea() to find the largest contour
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(np.array(area))
    cv2.drawContours(contour_img, contours, max_idx, 255, 2)
    '''
    plt.imshow(contour_img, 'gray')
    plt.title('max Contour')
    plt.show()
    #cv2.imshow("test", contour_img)
    #cv2.waitKey(0)
    return contour_img


def getFingerContourIntersections(contour_img, x) -> np.ndarray:
    '''
    Run along a column at position x, and return the 6 intersecting y-values with the finger contours.
    (For help check Palmprint_Algnment_Helper.pdf section 2b)
    :param contour_img:
    :param x: position of the image column to run along
    :return: y-values in np.ndarray in shape (6,)
    '''
    y = np.zeros(6)
    counter = -1
    white_counter = 1  # calculate the width of white => helps to find the middle value
    for i in range(1, len(contour_img)):
        if contour_img[i, x] == 255 and contour_img[i-1, x] == 255:  # if white->white
            white_counter = white_counter + 1
        elif contour_img[i-1, x] == 0 and contour_img[i, x] == 255:  # if black->white
            white_counter = 1  # reset white_counter
        elif contour_img[i-1, x] == 255 and contour_img[i, x] == 0:  # if white->black
            if (counter >= 0) and (counter <= 5):
                y[counter] = i - 1 - white_counter // 2
            counter = counter + 1  # calculate the times passing a white line
        # else do nothing
    return y


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
    if y1 == y2:
        # the same row
        for x in range(max(x1, x2), img.shape[1]):
            if img[y1, x] == 255:
                return y1, x
    if x1 == x2:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # the same colum
        for y in range(max(y1, y2), img.shape[0]):
            if img[y, x1] == 255:
                return y, x1
    # y1 = a * x1 + b, y2 = a * x2 + b => y1 - y2 = a * (x1 - x2)
    a = (y1 - y2) / (x1 - x2)
    b = y1 - a * x1
    for x in range(max(x1, x2), img.shape[1]):
        y = int(a * x + b)
        if img[y, x] == 255:
            return y, x

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
        slope2 = -1 / slope1
        b1 = k1[0] - slope1 * k1[1]
        b2 = k2[0] - slope2 * k2[1]
        centerX = (b2 - b1) / (slope1 - slope2)  # a1*x+b1=a2*x+b2 => 0=(a1-a2)*x+(b1-b2) => (b2-b1)/(a1-a2)=x
        centerY = centerX * slope1 + b1

    theta = np.arctan(slope2)

    M = cv2.getRotationMatrix2D((centerY, centerX), np.degrees(theta), scale=1.0)
    print(M)
    """
       M:
       [
       [cosA -sinA (1-cosA)*centerX+sinA*centerY]
       [sinA cosA  -sinA*centerX+(1-cosA)*centerY]
       ]
    """

    return M

def palmPrintAlignment(img):
    '''
    Transform a given image like in the paper using the helper functions above when possible
    :param img: greyscale image
    :return: transformed image
    '''

    # TODO threshold and blur
    blured = binarizeAndSmooth(img)

    # TODO find and draw largest contour in image
    max_contour = drawLargestContour(blured)
    # max_contour = max_contour.astype(np.uint8)
    # TODO choose two suitable columns and find 6 intersections with the finger's contour
    x1 = 10
    x2 = 30
    series1 = getFingerContourIntersections(max_contour, x1)
    series2 = getFingerContourIntersections(max_contour, x2)
    intersection_img = np.zeros(max_contour.shape)
    intersection_img = intersection_img + max_contour
    intersection_img[:, x1] = 255
    intersection_img[:, x2] = 255
    plt.imshow(intersection_img, cmap='gray')
    plt.title('line to check intersection possibilities')
    plt.show()

    # TODO compute middle points from these contour intersections
    middle_series1 = np.zeros(3).astype(int)
    middle_series2 = np.zeros(3).astype(int)
    print("??????: ", len(series1), len(series2))
    for i in range(3):
        middle_series1[i] = (series1[i*2] + series1[i*2+1]) // 2
        middle_series2[i] = (series2[i*2] + series2[i*2+1]) // 2

    # TODO extrapolate line to find k1-3
    k1 = findKPoints(max_contour, middle_series1[0], x1, middle_series2[0], x2)
    k2 = findKPoints(max_contour, middle_series1[1], x1, middle_series2[1], x2)
    k3 = findKPoints(max_contour, middle_series1[2], x1, middle_series2[2], x2)
    contour_img = drawCircle(max_contour, k1[1], k1[0])
    contour_img = drawCircle(contour_img, k2[1], k2[0])
    contour_img = drawCircle(contour_img, k3[1], k3[0])
    plt.imshow(contour_img, cmap='gray')
    plt.title('Contour with intersection')
    plt.show()

    # TODO calculate Rotation matrix from coordinate system spanned by k1-3
    m = getCoordinateTransform(k1, k2, k3)

    # TODO rotate the image around new origin
    a, b = np.shape(img)
    rotated = cv2.warpAffine(img, m, (b, a))

    return rotated
