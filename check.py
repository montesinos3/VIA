#!/usr/bin/env python

from platform import python_version
print(f'python .......... {python_version()}')

import numpy
print(f'numpy ........... {numpy.__version__}')

import matplotlib
print(f'matplotlib ...... {matplotlib.__version__}')

import cv2
print(f'OpenCV .......... {cv2.__version__}')

try:
    import sympy
    print(f'sympy ........... {sympy.__version__}')
except:
    print(f'sympy ........... NOT FOUND')

try:
    import dlib
    print(f'dlib ............ {dlib.__version__}')
except:
    print(f'dlib ............ NOT FOUND')

try:
    import face_recognition
    print(f'face_recognition  {face_recognition.__version__}')
except:
    print(f'face_recognition  NOT FOUND')

try:
    import torch
    sdev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'torch ........... {torch.__version__} {sdev}')
except:
    print(f'torch ........... NOT FOUND')

try:
    import os
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    print(f'keras ........... {keras.__version__}')
except:
    print(f'keras ........... NOT FOUND')

try:
    import jax
    print(f'jax ............. {jax.__version__} {jax.default_backend()} {jax.devices()}')
except:
    print(f'jax ............. NOT FOUND')

