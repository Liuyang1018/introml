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
    print(type(img[0,0]))
    print(type(IBinaryMap[0,0]))
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
    img = img.astype(np.uint8)
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
    plt.imshow(contour_img, 'gray')
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

    # a*x1+b=y1, a*x2+b=y2 => a*(x1-x2)=)y1-y2 => a=(y1-y2)/(x1-x2), b=y1-a*x1
    a = (y1 - y2) / (x1 - x2)
    b = y1 - a * x1

    # found = False
    print(y1)
    for y in range(y2 + 1, img.shape[0]):
        x = int((y - b) / a)
        if x >= img.shape[0] or x < 0:
            break
        if img[y, x] != img[y2, x2]:
            # found = True
            return tuple((y, x))
    # if not found:
    for y in range(y1-1, -1, -1):
        x = int((y - b) / a)
        if x >= img.shape[0] or x < 0:
            break
        if img[y, x] != img[y2, x2]:
            # found = True
            return tuple((y, x))
    # if not found:
    #    print("??????")


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
    x2 = 14
    serie1 = getFingerContourIntersections(max_contour, x1)
    serie2 = getFingerContourIntersections(max_contour, x2)

    # TODO compute middle points from these contour intersections
    middle_serie1 = np.zeros(3).astype(int)
    middle_serie2 = np.zeros(3).astype(int)
    for i in range(3):
        middle_serie1[i] = int((serie1[i*2] + serie1[i*2+1]) / 2)
        middle_serie2[i] = int((serie2[i*2] + serie2[i*2+1]) / 2)

    # TODO extrapolate line to find k1-3
    k1 = findKPoints(max_contour, middle_serie1[0], x1, middle_serie2[0], x2)
    k2 = findKPoints(max_contour, middle_serie1[1], x1, middle_serie2[1], x2)
    k3 = findKPoints(max_contour, middle_serie1[2], x1, middle_serie2[2], x2)
    contourimg = drawCircle(max_contour, k1[1], k1[0])
    contourimg = drawCircle(contourimg, k2[1], k2[0])
    contourimg = drawCircle(contourimg, k3[1], k3[0])
    plt.imshow(contourimg, cmap='gray')
    plt.show()

    # TODO calculate Rotation matrix from coordinate system spanned by k1-3
    m = getCoordinateTransform(k1, k2, k3)

    # TODO rotate the image around new origin
    a, b = np.shape(img)
    rotated = cv2.warpAffine(img, m, (b, a))

    return rotated
