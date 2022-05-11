from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm


def load_sample(filename, duration=4*44100, offset=44100//10):
    # Complete this function
    sound = np.load(filename)
    # plt.plot(sound)
    # plt.show()
    max_index = np.argmax(sound)
    start = max_index + offset
    return sound[start:start + duration]


def compute_frequency(signal, min_freq=20):
    # Complete this function
    fourier = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(fourier), d=1 / 44100)
    indexes = np.where(frequencies >= min_freq, frequencies, 0)
    index = np.argmax(np.where(indexes != 0, np.abs(fourier), 0))
    frequency = frequencies[index]
    return frequency


if __name__ == '__main__':
    # Implement the code to answer the questions here
    signal = load_sample(os.path.join('sounds', 'Piano.ff.XX.npy'))
    print(compute_frequency(signal))  # 1179.878
    # XX is D6? (1174)
# This will be helpful:
# https://en.wikipedia.org/wiki/Piano_key_frequencies
