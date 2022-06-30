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
    DR = 0
    k = np.shape(Rx)[0]
    for i in range(k):
        DR += np.abs(Rx[i] - Ry[i])

    DR /= k

    return DR



def calculate_Theta_Distance(Thetax, Thetay):
    '''
    calculate similarities of Fan features
    :param Thetax: Fan features of Person X
    :param Thetay: Fan features of Person Y
    :return: Similiarity index of the two feature vectors
    '''
    DT , lxx, lxy, lyy = 0, 0, 0, 0
    sumX, sumY = 0, 0
    k = np.shape(Thetax)[0]

    for i in range(k):
        sumX += Thetax[i]
        sumY += Thetay[i]
    sumX /= k
    sumY /= k

    for i in range(k):
        lxx += (Thetax[i] - sumX)**2
        lyy += (Thetay[i] - sumY)**2
        lxy += (Thetax[i] - sumX) * (Thetay[i] - sumY)

    DT = (1 - lxy**2 / (lxx * lyy)) * 100

    return DT
