import cv2
import numpy as np
import mediapipe as mp

# Inicializar el detector de manos de MediaPipe
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def precompute(image):
    """Extrae los puntos clave de la mano en la imagen modelo"""
    # Convertir a RGB para MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Detectar manos
    results = hands.process(rgb_image)
    
    if results.multi_hand_landmarks:
        # Extraer coordenadas de los puntos clave
        landmarks = results.multi_hand_landmarks[0].landmark
        points = np.array([(lm.x, lm.y) for lm in landmarks])
        
        # Normalizar puntos (centrar y escalar)
        centroid = np.mean(points, axis=0)
        points -= centroid
        scale = np.sqrt(np.sum(np.square(points)) / len(points))
        points /= scale
        
        return points
    return None

def procrustes_distance(points1, points2):
    """Calcula la distancia de Procrustes entre dos conjuntos de puntos"""
    if points1 is None or points2 is None:
        return float('inf')
    
    # Asegurarse de que los puntos están centrados
    points1 = points1 - np.mean(points1, axis=0)
    points2 = points2 - np.mean(points2, axis=0)
    
    # Normalizar
    norm1 = np.sqrt(np.sum(points1 ** 2))
    norm2 = np.sqrt(np.sum(points2 ** 2))
    
    if norm1 == 0 or norm2 == 0:
        return float('inf')
    
    points1 /= norm1
    points2 /= norm2
    
    # Calcular matriz de correlación
    corr = np.dot(points1.T, points2)
    
    # Descomposición SVD (descomposición en valores singulares)
    u, s, v = np.linalg.svd(corr) 
    # u es una matriz ortogonal de tamaño m x m(vectores singulares izquierdos).
    # s es una matriz diagonal de tamaño m x n con los valores singulares de corr
    # v es una matriz ortogonal de tamaño n x n(vectores singulares derechos).
    
    # Calcular matriz de rotación
    r = np.dot(u, v)
    
    # Aplicar rotación
    points1_rotated = np.dot(points1, r)
    
    # Calcular distancia euclidea
    distance = np.sum(np.square(points1_rotated - points2))
    
    # Convertir a similitud (mayor = más similar)
    similarity = 1.0 / (1.0 + distance)
    
    return similarity

def compare(image, model_features):
    """Compara los gestos de manos usando distancia de Procrustes"""
    if model_features is None:
        return 0.0
    #Reutilizo el metodo precompute para obtener los puntos de la imagen normalizados y centrados
    points = precompute(image)
    if points is None:
        return 0.0
    
    # Calcular similitud
    similarity = procrustes_distance(points, model_features)
    
    return similarity