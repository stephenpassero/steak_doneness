import cv2 as cv
import numpy as np

image = cv.imread('./train/1000_F_409785284_0SeCTRiQ0M5ASa4TlpDdrbsMIJSAvC0l_jpg.rf.20f26de85cca598302a0589e6d4071f2.jpg')
cv.imwrite('./test.jpg', image)

hsv_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)

red = np.uint8([[[0,0,240 ]]])
hsv_red = cv.cvtColor(red,cv.COLOR_BGR2HSV)

delta = 5
common = 125

# define range of red color in HSV
lower_red = np.array([common- delta, 0, 0])
upper_red = np.array([common + delta, 255, 255])

# Threshold the HSV image to get only red colors
mask = cv.inRange(hsv_image, lower_red, upper_red)

kernel = np.ones((3, 3), np.uint8)
opened_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)

close_kernel = np.ones((9, 9), np.uint8)
opened_closed_mask = cv.morphologyEx(opened_mask, cv.MORPH_CLOSE, close_kernel, iterations=2)

# Bitwise-AND mask and original image
result = cv.bitwise_and(image, image, mask=mask)
cv.imwrite('./output/result.jpg', result)

opened_result = cv.bitwise_and(image, image, mask=opened_mask)
cv.imwrite('./output/opened.jpg', opened_result)

opened_closed_result = cv.bitwise_and(image, image, mask=opened_closed_mask)
cv.imwrite('./output/opened_closed.jpg', opened_closed_result)


