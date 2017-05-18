#!/usr/bin/env python

# run.py
#
# Author(s):
#   Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
#
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import cv2
import fnmatch
import numpy as np
from shutil import rmtree

class NoImagesException(Exception):
  def __init__(self, msg):
    # call the parent class init function
    Exception.__init__(self, msg)

    # save the message of the exception
    self.msg = msg

class DirectoryNotFoundException(Exception):
  def __init__(self, msg):
    # call the parent class init function
    Exception.__init__(self, msg)

    # save the message of the exception
    self.msg = msg

def get_images(source_directory, **kwargs):
  # set default values for keyword arguments
  extensions = kwargs.get('extensions', ['jpg', 'png'])

  # get all images by iterating the valur of source_directory recursively
  images = []

  # throw some error when the source_directory does not exist
  if not os.path.exists(source_directory):
    raise DirectoryNotFoundException("Source directory not found: %s" % source_directory)

  # walk starting from the root of the value of source_directory until to the very last child of it
  # then get all files that matches the value of the extensions parameter
  for root, _, filenames in os.walk(source_directory):
    for extension in extensions:
      for filename in fnmatch.filter(filenames, '*.' + extension):
        images.append(os.path.join(root, filename))

  return images

# TODO: perform some checking on image size and resize them according to the size
#       800x450 with some aspect ratio of 16:9
def detect_skin():
  # boundaries of possible skin in HSV color space
  lower_boundary = np.array([0, 48, 120], dtype="uint8")
  upper_boundary = np.array([20, 255, 255], dtype="uint8")

  # resolve the path to the folder of source images and fetch all images
  directory = os.path.join(os.getcwd(), "assets/img/contribs")
  images = get_images(directory)

  # throw some error when the source_directory does not exist
  num_images = len(images)
  if num_images == 0:
    raise NoImagesException("No images in the source directory: %s" % directory)

  # resolve the the path to the output folder. if it does exist, remove it and recreate it
  out_directory = os.path.join(os.getcwd(), "out/detected-skins")
  if os.path.exists(out_directory):
    rmtree(out_directory)

  os.makedirs(out_directory)

  # convert all images to opencv matrices
  cv_im_instances = [{'path': image, 'inst': cv2.imread(image)} for image in images]

  for cv_im_idx, meta in enumerate(cv_im_instances):
    # destructure the dictionary
    inst = map(meta.get, ('inst'))

    # get the dimensions of the image
    rows, cols, _ = inst.shape

    # convert the image to HSV color space
    hsv = cv2.cvtColor(meta['inst'], cv2.COLOR_BGR2HSV)

    # create a binrary image mask based on the lower and upper boundaries of the color range
    blur = cv2.GaussianBlur(hsv, (3, 3), 0)
    mask = cv2.inRange(blur, lower_boundary, upper_boundary)

    # compute for the percentage of the possible detected skin
    percent = (float(cv2.countNonZero(mask)) / (rows * cols)) * 100

    # removing noise by ellipse structuring element using erosion and dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # get all contours present on the mask
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour_idx, contour in enumerate(contours):
      area = cv2.contourArea(contour)

      # draw borders on the detected contour if the area of the contour is greater than 1000
      if area > 1000:
        cv2.drawContours(inst, contours, contour_idx, (0, 255, 0), 2)

    # save the file to the filesystem
    filename = "%s/p%.2f-%s.jpg" % (out_directory, percent, str(cv_im_idx).rjust(4, '0'))
    cv2.imwrite(filename, inst)

if __name__ == "__main__":
  detect_skin()


