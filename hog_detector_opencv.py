#!usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os
import sys

params = sys.argv
hog = cv2.HOGDescriptor()
im = cv2.imread(params[1])
h = hog.compute(im)
print h.size    
# for y in h:
    # for x in y:
        # print x
    # print y
# print h
# def hog_detector(img):
#     # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     hog = cv2.HOGDescriptor()
#     hog_feature = hog.compute(img)
#     return hog_detector
#
# if __name__ == "__main__":
#     params = sys.argv
#     color_image = cv2.imread(params[1])
#     hog_feature = hog_detector(color_image)
#     print hog_detector
