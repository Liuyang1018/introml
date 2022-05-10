import numpy as np


def createChirpSignal(samplingrate: int, duration: int, freqfrom: int, freqto: int, linear: bool):
    # returns the chirp signal as list or 1D-array
    # TODO
    t = np.linspace(0, duration, samplingrate)  # time scaled
    if linear:
        c = (freqto - freqfrom)/duration  # chirp rate
        phase = 2 * np.pi * (freqfrom * t + 0.5 * c * t * t)
        print("linear phase", type(phase))
    else:
        k = (freqto / freqfrom) ** (1 / duration)  # the rate of exponential change in frequency
        phase = 2 * np.pi * freqfrom * ((k ** t - 1) / np.log(k))
        print("exponential phase", type(phase))
    return np.sin(phase)

