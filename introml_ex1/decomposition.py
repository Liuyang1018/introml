import numpy as np


def createTriangleSignal(samples: int, frequency: int, k_max: int):
    # returns the signal as 1D-array (np.ndarray)
    # TODO
    t = np.linspace(0, 1, samples).reshape((samples, 1))
    k = np.arange(k_max)
    to_sum_up = (-1) ** k * ((np.sin(2 * np.pi * (2 * k + 1) * frequency * t)) / ((2 * k + 1) ** 2))
    result = 8 / (np.pi ** 2) * np.sum(to_sum_up, axis=1)
    return result


def createSquareSignal(samples: int, frequency: int, k_max: int):
    # returns the signal as 1D-array (np.ndarray)
    # TODO
    t = np.linspace(0, 1, samples).reshape((samples, 1))
    k = np.arange(1, k_max)
    to_sum_up = (np.sin(2 * np.pi * (2 * k - 1) * frequency * t)) / (2 * k - 1)
    result = 4 / np.pi * np.sum(to_sum_up, axis=1)
    return result


def createSawtoothSignal(samples: int, frequency: int, k_max: int, amplitude: int):
    # returns the signal as 1D-array (np.ndarray)
    # TODO
    t = np.linspace(0, 1, samples).reshape((samples, 1))
    k = np.arange(1, k_max)
    to_sum_up = (np.sin(2 * np.pi * k * frequency * t)) / k
    result = amplitude / 2 - amplitude / np.pi * np.sum(to_sum_up, axis=1)
    return result
