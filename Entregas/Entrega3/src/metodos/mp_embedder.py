import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

from mediapipe.tasks.python.core.base_options import BaseOptions

# Obtener la ruta absoluta al directorio actual (methods)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Ruta al modelo TFLite que está en el mismo directorio
model_path = os.path.join(current_dir, 'embedder.tflite')

# Verificar que el archivo existe
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
    
print(f"Usando modelo desde: {model_path}")  # Para depuración

# Leer el contenido del archivo
with open(model_path, 'rb') as f:
    model_content = f.read()

# Inicializar el embedder de MediaPipe usando el contenido del modelo
base_options = BaseOptions(model_asset_buffer=model_content)
options = vision.ImageEmbedderOptions(base_options=base_options)

embedder = vision.ImageEmbedder.create_from_options(options)

def precompute(image):
    """Precomputa el embedding de la imagen usando MediaPipe"""
    # Convertir a formato RGB (MediaPipe usa RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Crear objeto de imagen de MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    # Obtener embedding
    embedding_result = embedder.embed(mp_image)
    # Devolver el vector de embedding
    return embedding_result.embeddings[0].embedding

def compare(image, model_features):
    """Compara la imagen con el modelo usando embeddings de MediaPipe"""
    # Convertir a formato RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Crear objeto de imagen de MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    # Obtener embedding
    embedding_result = embedder.embed(mp_image)
    image_embedding = embedding_result.embeddings[0].embedding
    
    # Calcular similitud de coseno
    dot_product = np.dot(model_features, image_embedding)
    norm_model = np.linalg.norm(model_features)
    norm_image = np.linalg.norm(image_embedding)
    similarity = dot_product / (norm_model * norm_image)
    
    return similarity