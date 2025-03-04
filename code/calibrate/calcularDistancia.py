import numpy as np

def calcular_distancia(K, punto1, punto2):
    """
    Calcula la distancia entre dos puntos seleccionados a partir de una matriz de dispersión K.

    Args:
    K (numpy.ndarray): Matriz de dispersión.
    punto1 (tuple): Coordenadas del primer punto (x1, y1).
    punto2 (tuple): Coordenadas del segundo punto (x2, y2).

    Returns:
    float: Distancia entre los dos puntos.
    """
    # Convertir los puntos a arrays de numpy
    p1 = np.array(punto1)
    p2 = np.array(punto2)
    
    # Calcular la diferencia entre los puntos
    diff = p1 - p2
    
    # Calcular la distancia usando la matriz de dispersión K
    distancia = np.sqrt(np.dot(np.dot(diff.T, K), diff))
    
    return distancia

# Ejemplo de uso
if __name__ == "__main__":
    K = np.array([[1, 0], [0, 1]])  # Matriz de dispersión de ejemplo
    punto1 = (1, 2)
    punto2 = (4, 6)
    
    distancia = calcular_distancia(K, punto1, punto2)
    print(f"La distancia entre los puntos {punto1} y {punto2} es: {distancia}")