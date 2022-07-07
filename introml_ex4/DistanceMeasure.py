'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''
import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!


def calculate_R_Distance(Rx, Ry):
    '''
    calculate similarities of Ring features
    :param Rx: Ring features of Person X
    :param Ry: Ring features of Person Y
    :return: Similiarity index of the two feature vectors
    '''
    DRxy = 1 / len(Rx) * np.sum(np.abs(Rx - Ry))
    return DRxy


def calculate_Theta_Distance(Thetax, Thetay):
    '''
    calculate similarities of Fan features
    :param Thetax: Fan features of Person X
    :param Thetay: Fan features of Person Y
    :return: Similiarity index of the two feature vectors
    '''
    k = len(Thetax)
    xx = Thetax - 1 / k * np.sum(Thetax)
    yy = Thetay - 1 / k * np.sum(Thetay)
    lxy = np.sum(xx * yy)
    lxx = np.sum(xx ** 2)
    lyy = np.sum(yy ** 2)
    DThetaxy = (1 - lxy * lxy / (lxx * lyy)) * 100
    return DThetaxy

