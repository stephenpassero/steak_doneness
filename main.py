import cv2 as cv
import numpy as np
import copy
image_names = [
  './train/1000_F_409785284_0SeCTRiQ0M5ASa4TlpDdrbsMIJSAvC0l_jpg.rf.20f26de85cca598302a0589e6d4071f2.jpg',
  './train/1000_F_66568719_Qau3aUyTW2yHcuYZVIN5tLR5aMPbBccS_jpg.rf.7574fb08346bfe66a782ae9becb6e7ae.jpg',
  './train/images-1-_jpg.rf.7b5743f39f7fa046dff13e1fdb3887b8.jpg',
  './train/images-6-_jpg.rf.3b1a69b709f5afe55d00f6e3c10b3d99.jpg',
  './train/images-13-_jpg.rf.50ac62b8f90243115b16de1a90bdaadf.jpg'
]

for i, img_name in enumerate(image_names):
  image = cv.imread(img_name)
  print(image.shape)

  cv.imwrite(f"./output/original{i}.jpg", image)

  # Note that we swap our red and blue channels here - when we mask based on the
  # color space, our red channel is now in the middle of our range
  hsv_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)

  intensity_channel = hsv_image[:,:,2]
  equalized_intensity = cv.equalizeHist(intensity_channel)
  hsv_image[:,:,2] = equalized_intensity

  equalized = cv.cvtColor(hsv_image, cv.COLOR_HSV2RGB)
  cv.imwrite(f"./output/equalized{i}.jpg", equalized)


  red = np.uint8([[[0, 0, 240]]])
  hsv_red = cv.cvtColor(red,cv.COLOR_BGR2HSV)

  delta = 11
  common = 125

  # define range of red color in HSV
  lower_red = np.array([common- delta, 75, 50])
  upper_red = np.array([common + delta, 255, 240])

  # Threshold the HSV image to get only red colors
  mask = cv.inRange(hsv_image, lower_red, upper_red)

  close_kernel = np.ones((9, 9), np.uint8)
  close_mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, close_kernel, iterations=2)

  open_kernel = np.ones((5, 5), np.uint8)
  opened_closed_mask = cv.morphologyEx(close_mask, cv.MORPH_OPEN, open_kernel, iterations=2)

  # Bitwise-AND the mask with the original image
  result = cv.bitwise_and(image, image, mask=mask)

  closed_result = cv.bitwise_and(image, image, mask=close_mask)

  opened_closed_result = cv.bitwise_and(image, image, mask=opened_closed_mask)
  cv.imwrite(f"./output/result{i}.jpg", opened_closed_result)