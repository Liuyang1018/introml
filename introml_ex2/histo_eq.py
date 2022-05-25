# Implement the histogram equalization in this file
import cv2
import numpy as np

# read the image as np.ndarray
img = cv2.imread('hello.png', cv2.IMREAD_GRAYSCALE)

# calculate the histogram manually (not sure if np.bincount is allowed, otherwise use for loop)
img_array = img.flatten()
hist = np.bincount(img_array)
padding = np.zeros(256-len(hist))
hist = np.append(hist, padding)
# alternative:
# hist = np.zeros(256)
# for i in range(256):
#     hist[i] = np.sum(img_array == i)

# check: the sum of the first 90 values should be 249
print(np.sum(hist[:90]) == 249)

# calculate the probability density function
probability = hist / np.sum(hist)

# calculate the cumulative distribution function (any alternative without loop?)
Cx = np.zeros(256)
for i in range(256):
    Cx[i] = np.sum(probability[:i+1])

# check: the sum of the first 90 values should begin with 0.001974977
print(np.sum(Cx[:90]) - 0.001974977 < 0.000000001)

# change the gray value of each pixel
Cmin = Cx[np.argmin(Cx)]
new_pixelvalue = np.floor((Cx - Cmin) / (1 - Cmin) * 255)
new_image = np.zeros(len(img_array)) + img_array
for i in range(256):
    new_image = np.where(img_array == i, new_pixelvalue[i], new_image)

# save the image as "kitty.png"
cv2.imwrite("kitty.png", new_image.reshape(img.shape))
