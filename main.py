import cv2 as cv
import numpy as np

image = cv.imread('./train/1000_F_409785284_0SeCTRiQ0M5ASa4TlpDdrbsMIJSAvC0l_jpg.rf.20f26de85cca598302a0589e6d4071f2.jpg')
cv.imwrite('./test.jpg', image)
hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

red = np.uint8([[[0,0,240 ]]])
hsv_red = cv.cvtColor(red,cv.COLOR_BGR2HSV)

# define range of red color in HSV
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

# Threshold the HSV image to get only blue colors
mask = cv.inRange(hsv_image, lower_red, upper_red)

cv.imwrite('./mask.jpg', mask)

# Bitwise-AND mask and original image
result = cv.bitwise_and(image, image, mask=mask)
cv.imwrite('./result.jpg', result)
