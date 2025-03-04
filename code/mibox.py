#!/usr/bin/env python

import cv2 as cv
from umucv.stream import autoStream
from umucv.util import putText, Video
import time
import numpy as np

cv.namedWindow("media")
RAD = [40]
cv.createTrackbar('radius', 'media', RAD[0], 200, lambda v: RAD.insert(0,v) ) 

cv.namedWindow("premask")
RAD2 = [45]
cv.createTrackbar('radius', 'premask', RAD2[0], 200, lambda v: RAD2.insert(0,v) ) 

cv.namedWindow("mask")
H = [0.06]
cv.createTrackbar('umbral', 'mask', int(H[0]*1000), 500, lambda v: H.insert(0,v/1000) )


video = Video()

    



for key,frame in autoStream():
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY).astype(float)/255
    cv.imshow('input',gray)

    media = cv.boxFilter(gray, -1, (RAD[0],RAD[0])) if RAD[0] > 0 else gray.copy()
    cv.imshow('media',media)

    dif = cv.absdiff(gray,media)
    cv.imshow('dif',dif)

    premask = cv.boxFilter(dif, -1, (RAD2[0],RAD2[0])) if RAD2[0] > 0 else dif.copy()
    cv.imshow('premask',premask*2)

    mask = premask > H[0]
    cv.imshow('mask',mask*gray)

    paragrabar=cv.cvtColor((mask*gray*255).astype('uint8'),cv.COLOR_GRAY2BGR)

    frames=np.hstack((frame,paragrabar))
    cv.imshow('Entrega',frames)
 
    video.write(frames, key, ord('v'))


video.release()

    #usar hstack para pegar imagenes y hacer un video de unos 15 segs