import cv2
import numpy as np

def precompute(image):
    """Precomputa el histograma de la imagen"""
    # Convertir a HSV para mejor comparación de colores
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Calcular histograma
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # Normalizar histograma
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def compare(image, model_features):
    """Compara la imagen con el modelo usando histogramas"""
    # Convertir a HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Calcular histograma
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # Normalizar
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    # Comparar histogramas (mayor valor = mayor similitud)
    similarity = cv2.compareHist(model_features, hist, cv2.HISTCMP_CORREL)
    return similarity