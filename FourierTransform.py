'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''

import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!



def polarToKart(shape, r, theta):
    '''
    convert polar coordinates with origin in image center to kartesian
    :param shape: shape of the image
    :param r: radius from image center
    :param theta: angle
    :return: y, x
    '''
    y = shape[0] / 2 + r * np.sin(theta)
    x = shape[1] / 2 + r * np.cos(theta)

    return y, x

def calculateMagnitudeSpectrum(img) -> np.ndarray:
    '''
    use the fft to generate a magnitude spectrum and shift it to the image center.
    Hint: This can be done with numpy :)
    :param img:
    :return:
    '''
    transformed = np.fft.fft2(img)
    shifted = np.fft.fftshift(transformed)
    mag = 20 * np.log10(1 + np.abs(shifted))

    return mag


def extractRingFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    '''
    Follow the approach to extract ring features
    :param magnitude_spectrum:
    :param k: number of rings to extract = #features
    :param sampling_steps: times to sample one ring
    :return: feature vector of k features
    '''
    rfeature = np.zeros((k,))
    sum = 0
    theta = np.linspace(0., np.pi, num = sampling_steps)

    for i in range(1, k + 1):
        for t in theta:
            for r in range(k * (i-1), k * i+1):
                y, x = polarToKart(np.shape(magnitude_spectrum), r, t)
                sum += magnitude_spectrum[int(y),int(x)]
        rfeature[i-1] = round(sum, 3)
        sum = 0

    return rfeature


def extractFanFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    """
    Follow the approach to extract Fan features
    Assume all rays have same length regardless of angle.
    Their length should be set by the smallest feasible ray.
    :param magnitude_spectrum:
    :param k: number of fans-like features to extract
    :param sampling_steps: number of rays to sample from in one fan-like area
    :return: feature vector of length k
    """
    tfeature = np.zeros((k,))
    sum = 0
    a, b = np.shape(magnitude_spectrum)
    #radius = np.linspace(0, min(a, b)/2-1, sampling_steps)
    for i in range(1, k + 1):
        for t in np.linspace(i-1, i, num = sampling_steps):
            for r in range(int(min(a, b)/2)-1):
                y, x = polarToKart(np.shape(magnitude_spectrum), r, t * np.pi / k)
                sum += magnitude_spectrum[int(y), int(x)]
        tfeature[i - 1] = round(sum, 3)
        sum = 0

    return tfeature



def calcuateFourierParameters(img, k, sampling_steps) -> (np.ndarray, np.ndarray):
    '''
    Extract Features in Fourier space following the paper.
    :param img: input image
    :param k: number of features to extract from each method
    :param sampling_steps: number of samples to accumulate for each feature
    :return: R, T feature vectors of length k
    '''
    magnitude = calculateMagnitudeSpectrum(img)
    R = extractRingFeatures(magnitude, k, sampling_steps)
    T = extractFanFeatures(magnitude, k, sampling_steps)

    return (R,T)
