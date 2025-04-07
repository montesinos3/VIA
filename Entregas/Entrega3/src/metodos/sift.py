import cv2 as cv

# Crear el detector SIFT y el matcher
sift = cv.SIFT_create(nfeatures=200)
matcher = cv.BFMatcher()

def precompute(image):
    """
    Precomputar las características (keypoints y descriptores) de una imagen usando SIFT.
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, mask=None)
    return {"keypoints": keypoints, "descriptors": descriptors}

def compare(frame, model_features):
    """
    Comparar un frame con un modelo precomputado usando SIFT.
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, mask=None)

    if descriptors is None or model_features["descriptors"] is None:
        return 0  # Si no hay descriptores, no hay coincidencia

    # Encontrar las dos mejores coincidencias para cada descriptor
    matches = matcher.knnMatch(descriptors, model_features["descriptors"], k=2)

    # Aplicar el "ratio test" para filtrar coincidencias
    good_matches = []
    for m in matches:
        if len(m) >= 2:
            best, second = m
            if best.distance < 0.75 * second.distance:
                good_matches.append(best)

    # Retornar el número de coincidencias como medida de similitud
    return len(good_matches)