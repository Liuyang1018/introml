import numpy as np

from chirp import createChirpSignal
from decomposition import createTriangleSignal, createSquareSignal, createSawtoothSignal

import matplotlib.pyplot as plt

# TODO: Test the functions imported in lines 1 and 2 of this file.
chirp_linear = createChirpSignal(200, 1, 1, 10, True)
chirp_exponential = createChirpSignal(200, 1, 1, 10, False)

if __name__ == '__main__':

    plt.plot(np.linspace(0, 1, 200), chirp_linear)
    plt.show()

    plt.plot(np.linspace(0, 1, 200), chirp_exponential)
    plt.show()