#!/usr/bin/env python

# > ./stream.py
# > ./stream.py --dev=help

import cv2          as cv
from umucv.stream import autoStream

for key,frame in autoStream():
    cv.imshow('input',255-frame)
    result = frame[:,:,[1,2,0]]
    cv.imshow('result',result)

cv.destroyAllWindows()

