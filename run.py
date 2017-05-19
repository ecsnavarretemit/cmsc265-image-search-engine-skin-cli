#!/usr/bin/env python

# run.py
#
# Author(s):
#   Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
#   Nelson Tejara <nhtejara@up.edu.ph>
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

class InvalidDimensionsException(Exception):
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

def validate_dimension(im, width, height):
  im_height, im_width, _ = im.shape

  return height == im_height and width == im_width

def detect_skin():
  # boundaries of possible skin in HSV color space
  lower_boundary = np.array([0, 48, 120], dtype="uint8")
  upper_boundary = np.array([20, 255, 255], dtype="uint8")

  # define image dimensions
  im_width = 800
  im_height = 450

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

  # get all images that contains invalid sizes
  invalid_images = filter(lambda meta: not validate_dimension(meta['inst'], im_width, im_height), cv_im_instances) # pylint: disable=W0110

  # throw some error when there are images that does not contain the right dimensions
  num_invalid_mages = len(invalid_images)
  if num_invalid_mages > 0:
    raise InvalidDimensionsException("Some of the images contains dimensions other than %sx%s" % (im_width, im_height))

  for cv_im_idx, meta in enumerate(cv_im_instances):
    # destructure the dictionary
    path, inst = map(meta.get, ('path', 'inst')) # pylint: disable=W0612

    # get the dimensions of the image
    rows, cols, _ = inst.shape

    # convert the image to HSV color space
    hsv = cv2.cvtColor(inst, cv2.COLOR_BGR2HSV)

    # create a binary image mask based on the lower and upper boundaries of the color range
    blur = cv2.GaussianBlur(hsv, (3, 3), 0)
    mask = cv2.inRange(blur, lower_boundary, upper_boundary)

    # removing noise by ellipse structuring element using erosion and dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # get all contours present on the mask
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # get only the contours greater than 1000 areas
    good_contours = filter(lambda contour: cv2.contourArea(contour) > 1000, contours) # pylint: disable=W0110

    # create a new mask based on the shape of the original mask detected
    # and draw highlight on the contours detected
    mask_bitmask = np.zeros(mask.shape, dtype=mask.dtype)
    for contour_idx, _ in enumerate(good_contours):
      cv2.drawContours(mask_bitmask, good_contours, contour_idx, (255, 255, 255), cv2.FILLED)
      cv2.drawContours(inst, good_contours, contour_idx, (0, 255, 0), 2)

    # remove the unnecessary contours on the original mask by performing
    # bitwise and operation using the original mask and the create bitmask
    bitwised_mask = cv2.bitwise_and(mask, mask_bitmask)

    # compute for the percentage of the possible detected skin
    percent = (float(cv2.countNonZero(bitwised_mask)) / (rows * cols)) * 100

    # save the file to the filesystem
    filename = "%s/p%.2f-%s.jpg" % (out_directory, percent, str(cv_im_idx).rjust(4, '0'))
    cv2.imwrite(filename, inst)

if __name__ == "__main__":
  detect_skin()


