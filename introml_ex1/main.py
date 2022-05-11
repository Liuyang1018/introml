from chirp import createChirpSignal
from decomposition import createTriangleSignal, createSquareSignal, createSawtoothSignal

import numpy as np
import matplotlib.pyplot as plt

# TODO: Test the functions imported in lines 1 and 2 of this file.

chirp_linear = createChirpSignal(200, 1, 1, 10, True)
chirp_exponential = createChirpSignal(200, 1, 1, 10, False)

decomposition_triangle = createTriangleSignal(200, 2, 10000)
decomposition_square = createSquareSignal(200, 2, 10000)
decomposition_sawtooth = createSawtoothSignal(200, 2, 10000, 1)

if __name__ == '__main__':

    t = np.linspace(0, 1, 200)

    plt.plot(t, chirp_linear)
    plt.show()

    plt.plot(t, chirp_exponential)
    plt.show()

    plt.plot(t, decomposition_triangle)
    plt.show()

    plt.plot(t, decomposition_square)
    plt.show()

    plt.plot(t, decomposition_sawtooth)
    plt.show()
