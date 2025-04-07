import argparse
import os
import cv2
from umucv.stream import autoStream
from umucv.util import putText
import importlib
import numpy as np
import time 

# Diccionario para almacenar los métodos de comparación disponibles
comparison_methods = {}

def load_comparison_methods():
    """Carga dinámicamente todos los métodos de comparación disponibles"""
    methods_dir = "metodos"
    for filename in os.listdir(methods_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]
            module = importlib.import_module(f"metodos.{module_name}")
            if hasattr(module, "compare") and hasattr(module, "precompute"):
                comparison_methods[module_name] = module
                print(f"Método cargado: {module_name}")

def main():
    parser = argparse.ArgumentParser(description="Aplicación de reconocimiento de imágenes")
    parser.add_argument("--models", required=True, help="Directorio con imágenes modelo")
    parser.add_argument("--method", required=True, help="Método de comparación a utilizar")
    parser.add_argument("--save", action="store_true", help="Habilitar guardado de nuevos modelos")
    args = parser.parse_args()
    
    # Cargar métodos de comparación disponibles
    load_comparison_methods()
    
    if args.method not in comparison_methods:
        print(f"Error: Método '{args.method}' no encontrado. Métodos disponibles: {list(comparison_methods.keys())}")
        return
    
    # Crear directorio de modelos si no existe
    if not os.path.exists(args.models):
        os.makedirs(args.models)
    
    # Cargar modelos (imagenes) desde el directorio
    models = {}
    for filename in os.listdir(args.models):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            model_path = os.path.join(args.models, filename)
            model_name = os.path.splitext(filename)[0]
            model_img = cv2.imread(model_path)
            if model_img is not None:
                # Precomputar características para el modelo
                models[model_name] = {
                    'image': model_img,
                    'features': comparison_methods[args.method].precompute(model_img)
                }
                print(f"Modelo cargado: {model_name}")
    
    # Variables para guardar nuevos modelos
    save_mode = False
    new_model_name = ""
    
    # Iniciar captura de video
    for key, frame in autoStream():
        # Mostrar frame original
        # cv2.imshow('Input', frame)
        
        
        # Activar modo de guardado
        if args.save and key == ord('s'):
            save_mode = True
            new_model_name = f"model_{int(time.time())}"
            continue

        # Modo de guardado de nuevo modelo (imagen)
        if save_mode:
            copia= frame.copy()
            putText(copia, f"Guardando modelo: {new_model_name} (Enter para confirmar, Esc para cancelar)")
            cv2.imshow('Save Model', copia)
            
            if key == 13:  # Enter
                model_path = os.path.join(args.models, f"{new_model_name}.jpg")
                cv2.imwrite(model_path, frame)
                models[new_model_name] = {
                    'image': frame.copy(),
                    'features': comparison_methods[args.method].precompute(frame)
                }
                print(f"Nuevo modelo guardado: {new_model_name}")
                save_mode = False
            elif key == 27:  # Esc
                print("Guardado cancelado.")
                save_mode = False
            continue
        
        
        # Si no hay modelos, mostrar mensaje
        if not models:
            putText(frame, "No hay modelos cargados. Añada modelos al directorio.")
            cv2.imshow('Result', frame)
            continue
            
        # Comparar frame con modelos
        start_time = time.time()
        results = {}
        for model_name, model_data in models.items():
            similarity = comparison_methods[args.method].compare(frame, model_data['features'])
            results[model_name] = similarity

        
        # Encontrar el mejor resultado
        best_match = max(results.items(), key=lambda x: x[1]) # como es una tupla (nombre,valor) toma como key el valor para compararlo
        processing_time = (time.time() - start_time) * 1000  # ms
        
        result_frame = frame.copy()
        if args.method != "sift":
            # Mostrarlo en pantalla
            putText(result_frame, f"Mejor coincidencia: {best_match[0]} ({best_match[1]:.2f})")
            putText(result_frame, f"Tiempo: {processing_time:.1f} ms", (10, 40))
            
            # Eliminar el mejor resultado para mostrar los demás
            results.pop(best_match[0])

            # Mostrar todos los resultados
            y_pos = 70
            for model_name, similarity in sorted(results.items(), key=lambda x: x[1], reverse=True):
                putText(result_frame, f"{model_name}: {similarity:.2f}", (10, y_pos))
                y_pos += 30
            
            # Muestro un mensaje para guardar un nuevo modelo en la esquina inferior derecha
            if args.save:
                putText(result_frame, "Presione 's' para guardar un nuevo modelo", (270, 350))

            cv2.imshow('Result', result_frame)
        else: 
            sift = cv2.SIFT_create(nfeatures=500)
            mejor_imagen=models[best_match[0]]['image']
            t0 = time.time()
            keypoints , descriptors = sift.detectAndCompute(result_frame, mask=None)
            t1 = time.time()
            putText(result_frame, f'{len(keypoints)} pts  {1000*(t1-t0):.0f} ms')
            
            k0,d0 = sift.detectAndCompute(mejor_imagen, mask=None)
            x0=mejor_imagen
            t2 = time.time()
            matches = cv2.BFMatcher().knnMatch(descriptors, d0, k=2)
            t3 = time.time()
            # Aplicar el "ratio test" para filtrar coincidencias
            good = []
            for m in matches:
                if len(m) >= 2:
                    best, second = m
                    if best.distance < 0.75 * second.distance:
                        good.append(best)

            if len(good) > 0:
                imgm = cv2.drawMatches(result_frame, keypoints, x0, k0, good,
                                    flags=0,
                                    matchColor=(128,255,128),
                                    singlePointColor=(128,128,128),
                                    outImg=None)
            else:
                print("No hay coincidencias buenas para dibujar.")
                imgm = result_frame.copy()
            putText(imgm ,f'{len(good)} matches  {1000*(t3-t2):.0f} ms', 
                      orig=(5,36), color=(200,255,200))   
            cv2.imshow('Result', imgm)
        
        
        

if __name__ == "__main__":
    main()
