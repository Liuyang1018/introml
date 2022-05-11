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

    A2 = load_sample(os.path.join('sounds', 'Piano.ff.A2.npy'))
    print('recorded A2:', compute_frequency(A2), '  A2:110.0000')

    A3 = load_sample(os.path.join('sounds', 'Piano.ff.A3.npy'))
    print('recorded A3:', compute_frequency(A3), '  A3:220.0000')

    A4 = load_sample(os.path.join('sounds', 'Piano.ff.A4.npy'))
    print('recorded A4:', compute_frequency(A4), '  A4:440.0000')

    A5 = load_sample(os.path.join('sounds', 'Piano.ff.A5.npy'))
    print('recorded A5:', compute_frequency(A5), '  A5:880.0000')

    A6 = load_sample(os.path.join('sounds', 'Piano.ff.A6.npy'))
    print('recorded A6:', compute_frequency(A6), '  A6:1760.000')

    A7 = load_sample(os.path.join('sounds', 'Piano.ff.A7.npy'))
    print('recorded A7:', compute_frequency(A7), '  A7:3520.000')

    XX = load_sample(os.path.join('sounds', 'Piano.ff.XX.npy'))
    print(compute_frequency(XX))  # 1179.878
    # XX is D6? (1174), because 883.5 < 1179.878 < 1760
    # ==> note is between A5 and A6, smaller frequency offset than for A6

    # Alice can recognize notes with low frequencies.
    # But she cannot correctly evaluate the notes with higher frequencies.
    # She recognizes notes with high frequencies as a lower one.
    # (e.g. A7: She set a 3610 and think it is 3520)

# This will be helpful:
# https://en.wikipedia.org/wiki/Piano_key_frequencies
