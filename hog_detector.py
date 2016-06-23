#!usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import sys
import math

params = sys.argv
im = cv2.imread(params[1])
print params[1]
gradientSize = 9

# gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
cellsize = [8,8]
# cv2.imshow("result", gray)
# cv2.imwrite("output_gray.png",gray)
# print im.shape
# print gray 
def hogDesc(grayimg):
    gradientMagnitude = []
    gradientOrientation = []
    for y in range(0, gray.shape[1] - 1):
        gradientMagnitude_row = []
        gradientOrientation_row = []
        for x in range(0, gray.shape[0] - 1):
            fx = 0
            fy = 0
            if x == 0:
                x1 = x
                x2 = x + 1
            elif x == gray.shape[0] -1:
                x1 = x - 1
                x2 = x
            else:
                x1 = x - 1
                x2 = x + 1

            if y == 0:
                y1 = y
                y2 = y + 1
            elif y == gray.shape[1] -1:
                y1 = y - 1
                y2 = y
            else:
                y1 = y - 1
                y2 = y + 1

            # print x1, x2, y1, y2
            fx = gray[x2][y] - gray[x1][y]
            fy = gray[x][y2] - gray[x][y1]

            gradientMagnitude_row.append( math.sqrt(fx**2 + fy**2) )
            Orientation = math.atan2(fy, (fx + 0.01))
            if Orientation < 0:
                gradientOrientation_row.append(math.pi + Orientation)
            else:
                gradientOrientation_row.append(Orientation)

        gradientMagnitude.append(gradientMagnitude_row)
        gradientOrientation.append(gradientOrientation_row)

    print gradientMagnitude
    print "\n\n Orientation\n\n"
    for gradientOrientation_row in gradientOrientation:
        for x in gradientOrientation_row:
            print x * 180 / math.pi

    # print gradientOrientation

    for i in range(0, gray.shape[0] / cellsize[0]):
        for j in range(0, gray.shape[1] / cellsize[1]):
            for k in range(cellsize[0]-1):
                for l in range(cellsize[1]-1):
                    x = i * cellsize[0] + k
                    y = j * cellsize[1] + l
                    m1 = (gradientOrientation[y][x] * 180 / math.pi) / (180 / gradientSize)
                    m2 = (gradientOrientation[y][x] * 180 / math.pi + (180/gradientSize)) / (180 / gradientSize)
                    linInt = ((gradientOrientation[y][x] * 180 / math.pi)-int(gradientOrientation[y][x] * 180 / math.pi))/float(180/9)
                    if linInt == 0:
                        m1 = m2

                    print "m1", m1, "m2", m2





if __name__ == "__main__":
    imgdata = cv2.imread(params[1])
    gray    = cv2.cvtColor(imgdata, cv2.COLOR_RGB2GRAY)
    hog_feature = hogDesc(gray)
    # print hog_feature
