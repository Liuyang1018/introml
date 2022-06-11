from PIL import Image
import numpy as np


def make_kernel(ksize, sigma):
    # implement the Gaussian kernel here
    kernel = np.zeros((ksize, ksize))
    middle = ksize // 2
    mini_mask = np.zeros((middle + 1, middle + 1))
    odd = True
    if ksize % 2 == 0:
        odd = False
    coeff = 1 / (2 * np.pi * sigma * sigma)
    for i in range(0, middle + 1):
        for j in range(i, middle + 1):
            x = middle - i
            y = middle - j
            mini_mask[i][j] = coeff * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            mini_mask[j][i] = mini_mask[i][j]
    bottom_left = np.flip(mini_mask, 0)
    upper_right = np.flip(mini_mask, 1)
    bottom_right = np.flip(mini_mask)

    kernel[0:middle+1, 0:middle+1] = mini_mask
    if odd:
        kernel[middle:, 0:middle+1] = bottom_left
        kernel[0:middle+1, middle:] = upper_right
        kernel[middle:, middle:] = bottom_right
    else:
        kernel[middle:, 0:middle+1] = bottom_left[:middle, :]
        kernel[0:middle+1, middle:] = upper_right[:, :middle]
        kernel[middle:, middle:] = bottom_right[:middle, :middle]
    """alternative, use for-loop to calculate the whole kernel:
    kernel = np.zeros((ksize, ksize))
    for i in range(0, ksize):
        for j in range(0, ksize):
            x = i - middle
            y = j - middle
            kernel[i][j] = coeff * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    """
    kernel = kernel / np.sum(kernel.flatten())

    return kernel


def slow_convolve(arr, k):
    # implement the convolution with padding here:

    # Initiate a new image with the same size as the input image
    m_new = np.zeros(arr.shape)

    # Pad your input image with zeros to the correct size (depends on the kernel)
    u, v = k.shape
    left = u//2
    top = v//2
    if u % 2 == 1:
        right = u//2
    else:
        right = (u-1)//2
    if v % 2 == 1:
        bottom = v//2
    else:
        bottom = (v-1)//2

    padding_shape = ((left, right), (top, bottom))
    m = np.pad(arr, padding_shape)
    # Go over each pixel of the new image and calculate the value for this pixel using the equation
    """"Alternative, flip the kernel first, using np.flip: """
    # k_flipped = np.flip(k)
    for i in range(left, m.shape[0] - right):
        for j in range(top, m.shape[1] - bottom):
            sum_up = 0
            for u_local in range(-right, left+1):
                for v_local in range(-bottom, top+1):
                    # without flip the kernel, calculate m from (2,2) to (0,0)
                    sum_up = sum_up + k[u_local + right][v_local + bottom] * m[i - u_local][j - v_local]
                    """"Alternative, flip the kernel first, using np.flip: """
                    """"Then Calculate m from (0,0) to (2,2)"""
                    # sum_up = sum_up + k_flipped[u_local + u // 2][v_local + v // 2] * m[i + u_local][j + v_local]
            m_new[i - left, j - top] = sum_up
    return m_new


if __name__ == '__main__':
    # todo: find better parameters
    for ksize in range(1, 11, 2):
        for sigma in range(1, 11, 2):
            k = make_kernel(ksize, sigma)
    
            # TODO: chose the image you prefer
            im = np.array(Image.open('input1.jpg'))
            # im = np.array(Image.open('input2.jpg'))
            # im = np.array(Image.open('input3.jpg'))
    
            # TODO: blur the image, subtract the result to the input,
            #       add the result to the input, clip the values to the
            #       range [0,255] (remember warme-up exercise?), convert
            #       the array to np.unit8, and save the result
            channels = im.shape[2]
            result = np.zeros(im.shape)
            for i in range(0, channels):
                result[:, :, i] = im[:, :, i] + (im[:, :, i] - slow_convolve(im[:, :, i], k))
            # clip
            result = np.where(result < 0, 0, result)
            result = np.where(result > 255, 255, result)

            # save
            image = Image.fromarray(np.uint8(result))
            filename = "result_ksize{0}_sigma{1}.png".format(ksize, sigma)
            image.save(filename)
    # ksize = 5, sigma = 3 looks not bad? I am blind...
