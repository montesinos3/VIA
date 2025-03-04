#!/usr/bin/env python

# https://github.com/googlesamples/mediapipe/blob/main/examples/image_segmentation/python/image_segmentation.ipynb

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from umucv.stream import autoStream
from umucv.util import check_and_download

import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from rembg import remove

check_and_download("deeplabv3.tflite","https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite")


# Create the options that will be used for ImageSegmenter
base_options = python.BaseOptions(
    model_asset_path='deeplabv3.tflite')

options = vision.ImageSegmenterOptions(
    base_options=base_options,
    output_category_mask=True)

segmenter = vision.ImageSegmenter.create_from_options(options)

for key, frame in autoStream():
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mpimage = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    segmentation_result = segmenter.segment(mpimage)
    category_mask = segmentation_result.category_mask.numpy_view()
    mask = np.expand_dims(category_mask>0,2)
    
    cv.imshow("Segmenter", frame * mask)

    cv.imshow('input',frame)
    if key == ord('r'):
        fx = 360/frame.shape[1]
        small = cv.resize(frame, (0,0), fx=fx, fy=fx)
        mask2 = cv.cvtColor(remove(small, only_mask=False), cv.COLOR_RGBA2RGB)
        cv.imshow("maskRembg",mask2)
        small_mask = cv.resize(frame * mask, (0,0), fx=fx, fy=fx)
        array = np.hstack((small, small_mask, mask2))
        cv.imshow("Stack", array)

