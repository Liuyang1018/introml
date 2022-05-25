import numpy as np
#
# NO OTHER IMPORTS ALLOWED
#


def create_greyscale_histogram(img):
    '''
    returns a histogram of the given image
    :param img: 2D image in greyscale [0, 255]
    :return: np.ndarray (256,) with absolute counts for each possible pixel value
    '''
    # TODO
    histogram, bin_edges = np.histogram(img, bins=256, range=(0, 256))
    return histogram


def binarize_threshold(img, t):
    '''
    binarize an image with a given threshold
    :param img: 2D image as ndarray
    :param t: int threshold value
    :return: np.ndarray binarized image with values in {0, 255}
    '''
    # TODO
    return (img > t) * 255


def p_helper(hist, theta: int):
    '''
    Compute p0 and p1 using the histogram and the current theta,
    do not take care of border cases in here
    :param hist:
    :param theta: current theta
    :return: p0, p1
    '''
    if np.sum(hist) == 0:
        probability = hist / np.finfo(float).eps
    else:
        probability = hist / np.sum(hist)
    p0 = np.sum(probability[0:theta + 1])
    p1 = np.sum(probability[theta + 1:])
    return p0, p1


def mu_helper(hist, theta, p0, p1):
    '''
    Compute mu0 and m1
    :param hist: histogram
    :param theta: current theta
    :param p0:
    :param p1:
    :return: mu0, mu1
    '''
    x = np.arange(0, len(hist))
    if p0 == 0:
        mu0 = 1 / np.finfo(float).eps * np.sum(x[0:theta+1] * hist[0:theta+1].T)
    else:
        mu0 = 1 / p0 * np.sum(x[0:theta+1] * hist[0:theta+1].T)

    if theta == len(hist) - 1:
        mu1 = 0
    else:
        if p1 == 0:
            mu1 = 1 / np.finfo(float).eps * np.sum(x[theta + 1:] * hist[theta + 1:].T)
        else:
            mu1 = 1 / p1 * np.sum(x[theta + 1:] * hist[theta + 1:].T)
    return mu0, mu1


def calculate_otsu_threshold(hist):
    '''
    calculates theta according to otsus method

    :param hist: 1D array
    :return: threshold (int)
    '''
    # TODO initialize all needed variables
    bins = len(hist)
    p0 = 0
    p1 = 0
    mu0 = 0
    mu1 = 0
    variance = np.zeros(bins)
    # TODO change the histogram, so that it visualizes the probability distribution of the pixels
    # --> sum(hist) = 1
    if np.sum(hist) == 0:
        hist = np.ones(bins) / bins
    else:
        hist = hist / np.sum(hist)

    print("SUM OF HIST:", np.sum(hist))
    # TODO loop through all possible thetas
    for theta in range(bins):
        # TODO compute p0 and p1 using the helper function
        p0, p1 = p_helper(hist, theta)
        # TODO compute mu and m1 using the helper function
        mu0, mu1 = mu_helper(hist, theta, p0, p1)
        # TODO compute variance
        variance[theta] = p0 * p1 * ((mu1 - mu0) ** 2)
        if theta == bins - 5:
            print("HIER:", p0, p1, mu0, mu1, variance[theta])
        # TODO update the threshold
    print("variances: ", variance)
    print(np.argmax(variance))
    return np.argmax(variance)


def otsu(img):
    '''
    calculates a binarized image using the otsu method.
    Hint: reuse the other methods
    :param image: grayscale image values in range [0, 255]
    :return: np.ndarray binarized image with values {0, 255}
    '''
    # TODO
    hist = create_greyscale_histogram(img)
    return binarize_threshold(img, calculate_otsu_threshold(hist))
