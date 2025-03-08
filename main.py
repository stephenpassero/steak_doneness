import cv2 as cv
import numpy as np

def equalize_histogram(image):
  hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

  intensity_channel = hsv_image[:,:,2]
  equalized_intensity = cv.equalizeHist(intensity_channel)
  hsv_image[:,:,2] = equalized_intensity

  return cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)

# Returns a mask defining regions of the image in the specific range
def mask_in_color_range(image, color_range, to_hsv_mapping = cv.COLOR_BGR2HSV):
  (range_low, range_high) = color_range

  hsv_image = cv.cvtColor(image, to_hsv_mapping)

  # Threshold the HSV image to get only colors in the range
  mask = cv.inRange(hsv_image, range_low, range_high)

  close_kernel = np.ones((9, 9), np.uint8)
  close_mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, close_kernel, iterations=2)

  open_kernel = np.ones((5, 5), np.uint8)
  return cv.morphologyEx(close_mask, cv.MORPH_OPEN, open_kernel, iterations=2)

def red_mask_for(image):
  delta = 11
  common = 125

  # define range of red color (technically blue, but we're going to flip red and blue channels) in HSV
  lower_red = np.array([common- delta, 95, 60])
  upper_red = np.array([common + delta, 255, 240])

  # Note that we swap our red and blue channels here - when we mask based on the
  # color space, our red channel is now in the middle of our range
  return mask_in_color_range(image, (lower_red, upper_red), cv.COLOR_RGB2HSV)

def brown_mask_for(image):
  delta = 8
  common = 110
  # define range of red color (technically blue, but we're going to flip red and blue channels) in HSV
  lower_red = np.array([common- delta, 20, 20])
  upper_red = np.array([common + delta, 175, 175])

  # Note that we swap our red and blue channels here - when we mask based on the
  # color space, our red channel is now in the middle of our range
  return mask_in_color_range(image, (lower_red, upper_red), cv.COLOR_RGB2HSV)

image_names = [
  './train/1000_F_409785284_0SeCTRiQ0M5ASa4TlpDdrbsMIJSAvC0l_jpg.rf.20f26de85cca598302a0589e6d4071f2.jpg',
  './train/1000_F_66568719_Qau3aUyTW2yHcuYZVIN5tLR5aMPbBccS_jpg.rf.7574fb08346bfe66a782ae9becb6e7ae.jpg',
  './train/images-1-_jpg.rf.7b5743f39f7fa046dff13e1fdb3887b8.jpg',
  './train/images-6-_jpg.rf.3b1a69b709f5afe55d00f6e3c10b3d99.jpg',
  './train/images-13-_jpg.rf.50ac62b8f90243115b16de1a90bdaadf.jpg',
  './train/images-89-_jpg.rf.ae52c350641719eb423abae1226cc544.jpg',
  './train/images-91-_jpg.rf.7cb33ca14f20805e9997e084e873349e.jpg',
  './train/images-102-_jpg.rf.e857834dffeedfc621ac9da285f175a6.jpg'
]

for i, img_name in enumerate(image_names):
  image = cv.imread(img_name)

  equalized = equalize_histogram(image)
  cv.imwrite(f"./output/equalized{i}.jpg", equalized)

  red_mask = red_mask_for(equalized)
  brown_mask = brown_mask_for(equalized)

  # Bitwise-AND the mask with the original image
  red_masked =  cv.bitwise_and(image, image, mask=red_mask)
  brown_masked =  cv.bitwise_and(image, image, mask=brown_mask)

  cv.imwrite(f"./output/red_mask_result{i}.jpg", red_masked)
  cv.imwrite(f"./output/brown_mask_result{i}.jpg", brown_masked)