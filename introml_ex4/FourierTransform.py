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
    return int(shape[0] / 2 + r * np.sin(theta)), int(shape[1] / 2 + r * np.cos(theta))


def calculateMagnitudeSpectrum(img) -> np.ndarray:
    '''
    use the fft to generate a magnitude spectrum and shift it to the image center.
    Hint: This can be done with numpy :)
    :param img:
    :return:
    '''
    f = np.fft.fft2(img)
    shifted = np.abs(np.fft.fftshift(f))
    print(np.max(np.abs(f)))
    # Reset the shifted image back into interval [0, 255]
    shifted_min = np.min(shifted)
    shifted_max = np.max(shifted)
    result = 255 / (shifted_max - shifted_min + np.finfo(float).eps) * (shifted - shifted_min)
    return result


def extractRingFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    '''
    Follow the approach to extract ring features
    :param magnitude_spectrum:
    :param k: number of rings to extract = #features
    :param sampling_steps: times to sample one ring
    :return: feature vector of k features
    '''
    features = np.zeros(k)
    for i in range(1, k + 1):
        for theta in np.linspace(0, np.pi, sampling_steps):
            for r in range(k * (i - 1), k * i + 1):
                y, x = polarToKart(magnitude_spectrum.shape, r, theta)
                features[i-1] = features[i-1] + magnitude_spectrum[y, x]
    return features


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
    features = np.zeros(k)
    for i in range(1, k + 1):
        for theta in np.linspace(i-1, i, sampling_steps):
            for r in range(0, k*k+1):
                y, x = polarToKart(magnitude_spectrum.shape, r, theta * np.pi / k)
                features[i-1] = features[i-1] + magnitude_spectrum[y, x]
    return features


def calcuateFourierParameters(img, k, sampling_steps) -> (np.ndarray, np.ndarray):
    '''
    Extract Features in Fourier space following the paper.
    :param img: input image
    :param k: number of features to extract from each method
    :param sampling_steps: number of samples to accumulate for each feature
    :return: R, T feature vectors of length k
    '''
    magnitude_spectrum = calculateMagnitudeSpectrum(img)
    R = extractRingFeatures(magnitude_spectrum, k, sampling_steps)
    T = extractFanFeatures(magnitude_spectrum, k, sampling_steps)
    return R, T
