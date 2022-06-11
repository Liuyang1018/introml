import matplotlib.pyplot as plt
from CannyEdgeDetector import canny
import cv2

img = cv2.imread('contrast.jpg', cv2.IMREAD_GRAYSCALE)
result = canny(img)

plt.imshow(result, 'gray')
plt.show()
# Q: Why do my results look different to those in the exercise sheet? Did they use other parameters?

