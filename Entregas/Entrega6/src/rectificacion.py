import cv2
import numpy as np
import argparse
import sys

# --- Variables Globales para Interacción ---
puntos_a_medir = []
imagen_a_pintar = None

def cargar_txt(ruta):
    """Carga los puntos de referencia desde un archivo de texto."""
    img_pts = []
    real_pts = []
    try:
        with open(ruta, 'r') as f:
            for linea in f:
                linea = linea.strip()
                if linea and not linea.startswith('#'):
                    parts = linea.split(",")
                    if len(parts) == 4:
                        try:
                            img_x, img_y, real_x, real_y = map(float, parts)
                            img_pts.append([img_x, img_y])
                            real_pts.append([real_x, real_y])
                        except ValueError:
                            print(f"Advertencia: Ignorando línea mal formada: {linea}")
                    else:
                        print(f"Advertencia: Ignorando línea con número incorrecto de valores: {linea}")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de referencias: {ruta}")
        return None, None
    except Exception as e:
        print(f"Error leyendo el archivo de referencias: {e}")
        return None, None

    if len(img_pts) < 4:
        print("Error: Se necesitan al menos 4 puntos de referencia.")
        return None, None

    # Convertir a np.arrays
    img_final = np.array(img_pts, dtype=np.float32)
    real_final = np.array(real_pts, dtype=np.float32)

    return img_final, real_final

def seleccionar_punto(evento, x, y, flags, param):
    """Callback del ratón para seleccionar puntos."""
    global puntos_a_medir, imagen_a_pintar
    
    if evento == cv2.EVENT_LBUTTONDOWN:
        if len(puntos_a_medir) < 2:
            puntos_a_medir.append((x, y))
            # Dibujar punto en la copia de la imagen
            cv2.circle(imagen_a_pintar, (x, y), 5, (203, 192, 255), -1)
            cv2.circle(imagen_a_pintar, (x, y), 6, (0, 0, 0), 1) # Borde negro
            cv2.imshow("Rectificación de Perspectiva - Selección de Puntos", imagen_a_pintar)
            print(f"Punto {len(puntos_a_medir)} seleccionado: ({x}, {y})")
        else:
            print("Ya has seleccionado 2 puntos. Pulsa 'r' para reiniciar.")

def main(ruta_imagen, ruta_txt):
    global puntos_a_medir, imagen_a_pintar

    # Cargar imagen original
    img_original = cv2.imread(ruta_imagen)
    if img_original is None:
        print(f"Error: No se pudo cargar la imagen: {ruta_imagen}")
        return

    # Cargar puntos de referencia
    pts_img, pts_real = cargar_txt(ruta_txt)
    if pts_img is None or pts_real is None:
        return

    print(f"Cargados {len(pts_img)} puntos de referencia.")

    # Calcular homografía
    homografia, _ = cv2.findHomography(pts_img, pts_real, cv2.RANSAC, 5.0)
    if homografia is None:
        print("Error: No se pudo calcular la homografía. Verifica los puntos de referencia.")
        return

    # Preparar para interacción
    imagen_a_pintar = img_original.copy()
    nombre_frame = "Rectificación de Perspectiva - Selección de Puntos"
    cv2.namedWindow(nombre_frame)
    cv2.setMouseCallback(nombre_frame, seleccionar_punto)

    print("\n------ Ayuda ------")
    print("Pulsa 'r' para reiniciar la selección de puntos o 'q' para salir.")
    print("Haz click en dos puntos en la imagen para medir la distancia real entre ellos.")
    print("Los puntos seleccionados se dibujarán en la imagen.")
    print("---------------------\n")

    while True:

        cv2.imshow(nombre_frame, imagen_a_pintar)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Reiniciando puntos.")
            puntos_a_medir = []
            imagen_a_pintar = img_original.copy() # Restaurar imagen limpia

        # Si se han seleccionado dos puntos
        if len(puntos_a_medir) == 2:
            a, b = puntos_a_medir # Descomponer los puntos seleccionados en los puntos a y b

            array_puntos_imagen = np.array([[a, b]], dtype=np.float32)

            array_puntos_real = cv2.perspectiveTransform(array_puntos_imagen, homografia)

            if array_puntos_real is None:
                 print("Transformación de perspectiva incompleta, error al tranformar.")
                 puntos_a_medir = [] 
                 imagen_a_pintar = img_original.copy()
                 continue

            a_real = array_puntos_real[0, 0] 
            b_real = array_puntos_real[0, 1] 

            distancia = np.linalg.norm(a_real - b_real) # Distancia en centímetros

            cv2.line(imagen_a_pintar, a, b, (255, 0, 0), 2)
            
            x_texto = int((a[0] + b[0]) / 2)
            y_texto = int((a[1] + b[1]) / 2) - 10 # Un poco arriba
            texto = f"{distancia:.2f}cm" # Asume unidades de referencias.txt
            cv2.putText(imagen_a_pintar, texto, (x_texto, y_texto),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA) # Borde blanco
            cv2.putText(imagen_a_pintar, texto, (x_texto, y_texto),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA) # Texto negro

            print(f"Distancia entre los dos puntos: {distancia:.2f} centímetros")

            # Muestro resultado final
            cv2.imshow(nombre_frame, imagen_a_pintar)

            # Limpio los puntos para poder seleccionar un nuevo par
            puntos_a_medir = []

            print("Pulsa 'r' para reiniciar los puntos, 'q' para salir o selecciona 2 nuevos puntos.")

    cv2.destroyAllWindows()
    print("Programa finalizado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Ruta a la imagen de entrada (ej. ../images/imagen.jpg)')
    parser.add_argument('puntos', help='Ruta al txt con los puntos de referencia (ej. "./points.txt") (formato: pixeles_x pixeles_y cm_x cm_y)') 
    args = parser.parse_args()

    main(args.image, args.puntos)