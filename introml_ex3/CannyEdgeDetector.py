import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import convo

#
# NO MORE MODULES ALLOWED
#


def gaussFilter(img_in, ksize, sigma):
    """
    filter the image with a gauss kernel
    :param img_in: 2D greyscale image (np.ndarray)
    :param ksize: kernel size (int)
    :param sigma: sigma (float)
    :return: (kernel, filtered) kernel and gaussian filtered image (both np.ndarray)
    """
    # TODO
    k = convo.make_kernel(ksize, sigma)
    return k, convolve(img_in, k, output=int)  # I don't understand, why int?


def sobel(img_in):
    """
    applies the sobel filters to the input image
    Watch out! scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    """
    # TODO
    # sobel filters
    g_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    g_y = np.rot90(g_x)
    print(g_y)  # Q: I don't understand the flip here... It seems to be upside-down, but not 180 grad???
    # return sobel filtered images in x- and y-direction
    return convolve(img_in, g_x, output=int), convolve(img_in, g_y, output=int)


def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    # TODO
    g = np.sqrt(gx ** 2 + gy ** 2)
    theta = np.arctan2(gy, gx)
    return g.astype(np.int), theta


def convertAngle(angle):
    """
    compute nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
    # TODO
    grad = (angle / 2 / np.pi * 360) % 180
    if (grad >= 112.5) and (grad < 157.5):
        return 135
    elif (grad >= 77.5) and (grad < 112.5):
        return 90
    elif (grad >= 22.5) and (grad < 77.5):
        return 45
    else:
        return 0


def maxSuppress(g, theta):
    """
    calculate maximum suppression
    :param g:  (np.ndarray)
    :param theta: 2d image (np.ndarray)
    :return: max_sup (np.ndarray)
    """
    # TODO Hint: For 2.3.1 and 2 use the helper method above
    max_sup = np.zeros(g.shape)
    im_shape = theta.shape
    for i in range(1, im_shape[0] - 1):
        for j in range(1, im_shape[1] - 1):
            angle = convertAngle(theta[i][j])
            # Q: Example in the exercise sheet is wrong? angle == 0 checks left and right, == 90 checks up and down
            if angle == 0:
                if (g[i][j] >= g[i][j+1]) and (g[i][j] >= g[i][j-1]):
                    max_sup[i][j] = g[i][j]
            elif angle == 45:
                if (g[i][j] >= g[i+1][j-1]) and (g[i][j] >= g[i-1][j+1]):
                    max_sup[i][j] = g[i][j]
            if angle == 90:
                if (g[i][j] >= g[i+1][j]) and (g[i][j] >= g[i-1][j]):
                    max_sup[i][j] = g[i][j]
            if angle == 135:
                if (g[i][j] >= g[i+1][j+1]) and (g[i][j] >= g[i-1][j-1]):
                    max_sup[i][j] = g[i][j]
    return max_sup


def hysteris(max_sup, t_low, t_high):
    """
    calculate hysteris thresholding.
    Attention! This is a simplified version of the lectures hysteresis.
    Please refer to the definition in the instruction

    :param max_sup: 2d image (np.ndarray)
    :param t_low: (int)
    :param t_high: (int)
    :return: hysteris thresholded image (np.ndarray)
    """
    # TODO
    threshimg = np.ones(max_sup.shape)
    threshimg = np.where(max_sup <= t_low, 0, threshimg)
    threshimg = np.where(max_sup > t_high, 2, threshimg)
    padded = np.pad(threshimg, ((1, 1), (1, 1)))

    for i in range(1, padded.shape[0]-1):
        for j in range(1, padded.shape[1]-1):
            if padded[i][j] == 2:
                padded[i-1:i+2, j-1:j+2] = np.where(padded[i-1:i+2, j-1:j+2] == 1, 255, padded[i-1:i+2, j-1:j+2])
                padded[i][j] = 255
    return padded[1:padded.shape[0], 1:padded.shape[1]]


def canny(img):
    # gaussian
    kernel, gauss = gaussFilter(img, 5, 2)

    # sobel
    gx, gy = sobel(gauss)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()
    plt.show()

    # gradient directions
    g, theta = gradientAndDirection(gx, gy)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()

    # maximum suppression
    maxS_img = maxSuppress(g, theta)

    # plotting
    plt.imshow(maxS_img, 'gray')
    plt.show()

    result = hysteris(maxS_img, 50, 75)

    return result
