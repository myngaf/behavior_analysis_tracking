import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

def callback(x):
    print(x)

# read image
bg_path = r'D:\Experiments\PC_induction_\backgrounds'
images = [f for f in listdir(bg_path) if isfile(join(bg_path, f))]
n = 1
max_images = len(images)
img = cv2.imread(join(bg_path, images[n])) #read image as grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# bg is the image used for processing
blur = 5
bg = cv2.medianBlur(gray, blur)
input = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)

# parameters
h, w = bg.shape
dp, mindist, param1, param2 = 1, 250, 200, 50

# output is the output image
# output = np.zeros((h, w), dtype=np.uint8)
canny = cv2.Canny(img, param1/2, param1)
output = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)

# circle algorithm
circles = cv2.HoughCircles(bg,
                           cv2.HOUGH_GRADIENT,
                           dp,
                           mindist,
                           param1=param1,
                           param2=param2,
                           minRadius=int(np.round(h / 4 * 1.8)),
                           maxRadius=int(np.round(h / 2 * 1.1)))

# modify output image
if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0]:
        x, y, r = circle
        cv2.circle(output, (x, y), r, (0, 0, 255), thickness=1)

# create widget
cv2.namedWindow('image', cv2.WINDOW_NORMAL) # make a window with name 'image'

cv2.createTrackbar('#image', 'image', n, max_images-1, callback)
cv2.createTrackbar('blur', 'image', blur, 200, callback)
cv2.createTrackbar('dp', 'image', dp, 20, callback)
cv2.createTrackbar('minDist', 'image', mindist, 1024, callback)
cv2.createTrackbar('param1', 'image', param1, 512, callback) #lower threshold trackbar for window 'image
cv2.createTrackbar('param2', 'image', param2, 512, callback) #upper threshold trackbar for window 'image

cv2.setTrackbarMin('#image', 'image', 1)
cv2.setTrackbarMin('blur', 'image', 1)
cv2.setTrackbarMin('dp', 'image', 1)
cv2.setTrackbarMin('minDist', 'image', 1)
cv2.setTrackbarMin('param1', 'image', 1)
cv2.setTrackbarMin('param2', 'image', 1)

while(1):
    numpy_horizontal_concat = np.concatenate((input, output), axis=1) # to display image side by side
    # r = 50.0 / img.shape[0]
    # dim = (int(img.shape[1] * r), 50)
    # resized = cv2.resize(numpy_horizontal_concat, dim)
    cv2.resizeWindow('image', 1400, 700)
    cv2.imshow('image', numpy_horizontal_concat)

    k = cv2.waitKey(1) & 0xFF
    if k == 27: #escape key
        break

    n = cv2.getTrackbarPos('#image', 'image')
    blur = cv2.getTrackbarPos('blur', 'image')
    if (blur % 2) == 0:
        blur += 1
    dp = cv2.getTrackbarPos('dp', 'image')
    mindist = cv2.getTrackbarPos('minDist', 'image')
    param1 = cv2.getTrackbarPos('param1', 'image')
    param2 = cv2.getTrackbarPos('param2', 'image')

    img = cv2.imread(join(bg_path, images[n]))  # read image as grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # bg = cv2.GaussianBlur(gray, (blur,blur), 0)
    # bg = cv2.bilateralFilter(gray, blur, 20, 20)
    bg = cv2.medianBlur(gray, blur)
    input = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)

    # output = np.zeros((h, w), dtype=np.uint8)
    canny = cv2.Canny(bg, param1 / 2, param1)
    output = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)

    circles = cv2.HoughCircles(bg,
                               cv2.HOUGH_GRADIENT,
                               dp,
                               mindist,
                               param1 = param1,
                               param2 = param2,
                               minRadius=int(np.round(h / 4 * 1.8)),
                               maxRadius=int(np.round(h / 2 * 1.1)))

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0]:
            x, y, r = circle
            cv2.circle(output, (x, y), r, (0, 0, 255), thickness=1)

cv2.destroyAllWindows()