import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = 'image.jpg' 
original_image = cv2.imread(image_path)

rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
inverted_image = cv2.bitwise_not(gray_image)
blurred_image = cv2.GaussianBlur(inverted_image, (21, 21), sigmaX=0, sigmaY=0)
inverted_blurred_image = cv2.bitwise_not(blurred_image)
pencil_sketch = cv2.divide(gray_image, inverted_blurred_image, scale=256.0)

plt.figure(figsize=(14, 8))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(rgb_image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Pencil Sketch")
plt.imshow(pencil_sketch, cmap='gray')
plt.axis('off')

plt.show()