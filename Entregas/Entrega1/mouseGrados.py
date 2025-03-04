#!/usr/bin/env python

import cv2 as cv
from umucv.stream import autoStream
from collections import deque #lista doblemente enlazada, es decir, se puede acceder a los elementos de la lista desde el principio y el final
from umucv.util import putText
import numpy as np

puntos = deque(maxlen=2) #maxlen=2 para que solo se guarden los ultimos 2 puntos
#z=3
#K=np.array([[1,0],[0,1]]) #matriz de dispersión 

def manejador(event, x, y, flags, param):
    #global z para poder modificar la variable global
    if event == cv.EVENT_LBUTTONDOWN:
        print(x,y)
        #z=5
        puntos.append([x,y])

cv.namedWindow("webcam")
cv.setMouseCallback("webcam", manejador)


for key, frame in autoStream():
    for p in puntos:
        cv.circle(frame, p, 5, (0,0,255), thickness=-1) # (frame, tupla de puntos, radio, color, grosor)
    if len(puntos)==2:
        cv.line(frame, puntos[0], puntos[1], (0,0,255), 2) # (frame, punto1, punto2, color, grosor)
        #cv.line(frame, *puntos, (0,0,255), 2) # el * es para pasarle todos los puntos a la vez
        #xm, ym = np.mean(puntos, axis=0) #hace la media por columnas gracias al axis=0, para asi tener la media de x y la media de y
        pm = np.mean(puntos, axis=0).astype(int) #para tener el par de coordenadas en la variable pm (punto medio)
        #putText(frame, "Hola", (int(xm),int(ym)))
        #d=np.linalg.norm(np.diff(puntos)) #diff no lo hace bien
        d=np.linalg.norm(np.array(puntos[0])-puntos[1]) #distancia entre los dos puntos
        #distancia = np.sqrt(np.dot(np.dot(d.T, K), d))
        
        fov=65.97
        grados=fov*d/1920
        # resta=np.array(puntos[0])-puntos[1]
        # grados=(np.arctan2(resta[1], resta[0]))*57,2958
        # grados=np.arctan(d/1479)
        # deg=d/1479
        msg=f"{grados} grados"
        putText(frame, msg, pm)
        #putText(frame, f"{deg}")

    cv.imshow('webcam',frame)
    #print(z)

cv.destroyAllWindows()

