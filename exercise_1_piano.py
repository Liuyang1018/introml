from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def load_sample(filename, duration=4*44100, offset=44100//10):
    # Complete this function
    sounds = np.load(filename)
    highest_point = np.argmax(sounds)
    start_point = highest_point + offset
    return sounds[start_point:start_point + duration]

def compute_frequency(signal, min_freq=20):
    # Complete this function
    ft = abs(np.fft.fft(signal))
    freq = np.fft.fftfreq(signal.size, d=1.0 / 44100.0)
    new_f = np.copy(freq)
    new_f[np.where(freq < min_freq)] = 0
    peak = np.argmax(ft)
    return ceil(new_f[peak])

if __name__ == '__main__':
# Implement the code to answer the questions here
  #plt.plot(load_sample('sounds/Piano.ff.A7.npy'))
  #plt.show()
  compute_frequency(load_sample('sounds/Piano.ff.A7.npy'))

# This will be helpful:
# https://en.wikipedia.org/wiki/Piano_key_frequencies
