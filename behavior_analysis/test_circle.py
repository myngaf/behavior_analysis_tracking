import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import cv2
from skimage import io, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

mpl.use('TkAgg')

dir = r'F:\Martin\PC_induction\backgrounds\2024032901.tiff'
input = io.imread(dir)



background = cv2.imread(str(dir), cv2.IMREAD_GRAYSCALE)
print(dir)
background_blur = cv2.medianBlur(background, 5)
hist = cv2.calcHist([background_blur], [0], None, [256], [0, 256])
th=225
edges = cv2.Canny(background_blur, np.round(th/2), th)


# Load picture and detect edges
image = img_as_ubyte(input)
th = 50
edges = canny(image, sigma=3, low_threshold=np.round(th/2), high_threshold=th)

# Show image
plt.imshow(edges)
plt.show()

plt.plot(hist)
plt.show()


# Detect two radii
hough_radii = np.arange(450, 550, 2)
hough_res = hough_circle(edges, hough_radii)

# Select the most prominent 3 circles
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=3)

# Draw them
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
image = color.gray2rgb(image)
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
    image[circy, circx] = (220, 20, 20)

ax.imshow(image, cmap=plt.cm.gray)
plt.show()
