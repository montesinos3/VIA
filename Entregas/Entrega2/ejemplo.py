import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import argparse
import os

def convolution_from_scratch(image, kernel):
    """
    Perform convolution of an image with a kernel using loops.
    :param image: 2D numpy array representing the image
    :param kernel: 2D numpy array representing the kernel (mask)
    :return: 2D numpy array of the convolved image
    """
    # Verificar si la imagen tiene canales de color
    if len(image.shape) == 3:
        height, width, channels = image.shape
        output = np.zeros_like(image)
        
        for c in range(channels):
            kernel_height, kernel_width = kernel.shape
            
            # Calculate padding size
            pad_height = kernel_height // 2
            pad_width = kernel_width // 2
            
            # Pad the image with zeros
            padded_image = np.pad(image[:,:,c], ((pad_height, pad_height), (pad_width, pad_width)), 
                                 mode='constant', constant_values=0)
            
            # Perform convolution using loops
            for i in range(height):
                for j in range(width):
                    # Extract the region of interest
                    region = padded_image[i:i + kernel_height, j:j + kernel_width]
                    # Perform element-wise multiplication and sum the result
                    output[i, j, c] = np.sum(region * kernel)
        
        return output
    else:
        # Para imágenes en escala de grises
        kernel_height, kernel_width = kernel.shape
        image_height, image_width = image.shape

        # Calculate padding size
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2

        # Pad the image with zeros
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), 
                             mode='constant', constant_values=0)

        # Initialize the output image
        output = np.zeros_like(image)

        # Perform convolution using loops
        for i in range(image_height):
            for j in range(image_width):
                # Extract the region of interest
                region = padded_image[i:i + kernel_height, j:j + kernel_width]
                # Perform element-wise multiplication and sum the result
                output[i, j] = np.sum(region * kernel)

        return output

def convolution_with_opencv(image, kernel):
    """
    Perform convolution of an image with a kernel using OpenCV's filter2D function.
    :param image: 2D or 3D numpy array representing the image
    :param kernel: 2D numpy array representing the kernel (mask)
    :return: numpy array of the convolved image
    """
    return cv2.filter2D(image, -1, kernel)

def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Aplicar convolución a una imagen.')
    parser.add_argument('image_path', type=str, help='Ruta a la imagen de entrada')
    parser.add_argument('--kernel', type=str, default='blur', 
                        choices=['blur', 'edge_h', 'edge_v', 'sharpen', 'emboss'],
                        help='Tipo de kernel a aplicar (predeterminado: blur)')
    parser.add_argument('--save', action='store_true', help='Guardar las imágenes resultantes')
    parser.add_argument('--output_dir', type=str, default='output', 
                        help='Directorio para guardar las imágenes resultantes')
    
    args = parser.parse_args()
    
    # Verificar si la imagen existe
    if not os.path.isfile(args.image_path):
        print(f"Error: No se encontró la imagen en {args.image_path}")
        return
    
    # Cargar la imagen
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen desde {args.image_path}")
        return
    
    # Convertir de BGR a RGB para visualización con matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convertir a escala de grises para simplificar la convolución
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Definir kernels
    kernels = {
        'blur': np.ones((3, 3), dtype=np.float32) / 9,
        'edge_h': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32),
        'edge_v': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32),
        'sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
        'emboss': np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    }
    
    kernel = kernels[args.kernel]
    
    print(f"Aplicando kernel {args.kernel} a la imagen...")
    
    # Medir tiempo de ejecución de la implementación desde cero
    start_time = time.time()
    result_scratch = convolution_from_scratch(image_gray, np.ones((3, 3), dtype=np.float32) / 9)
    end_time = time.time()
    execution_time_scratch = end_time - start_time
    print(f"Tiempo de ejecución (implementación desde cero): {execution_time_scratch:.6f} segundos")
    
    # Medir tiempo de ejecución de OpenCV
    start_time = time.time()
    result_opencv = convolution_with_opencv(image_gray, kernel)
    end_time = time.time()
    execution_time_opencv = end_time - start_time
    print(f"Tiempo de ejecución (OpenCV): {execution_time_opencv:.6f} segundos")
    
    # Comparación de eficiencia
    speedup = execution_time_scratch / execution_time_opencv if execution_time_opencv > 0 else float('inf')
    print(f"OpenCV es aproximadamente {speedup:.2f} veces más rápido")
    
    # Mostrar los resultados
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Imagen Original")
    plt.imshow(image_rgb)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title(f"Convolución desde cero\n{execution_time_scratch:.6f}s")
    plt.imshow(result_scratch, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title(f"Convolución OpenCV\n{execution_time_opencv:.6f}s")
    plt.imshow(result_opencv, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Guardar resultados si se solicita
    if args.save:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        base_filename = os.path.splitext(os.path.basename(args.image_path))[0]
        
        plt.savefig(f"{args.output_dir}/{base_filename}_{args.kernel}_comparison.png")
        cv2.imwrite(f"{args.output_dir}/{base_filename}_{args.kernel}_scratch.png", result_scratch)
        cv2.imwrite(f"{args.output_dir}/{base_filename}_{args.kernel}_opencv.png", result_opencv)
        print(f"Imágenes guardadas en el directorio {args.output_dir}")
    
    plt.show()
    
    # Verificar que los resultados son similares
    difference = np.mean(np.abs(result_scratch - result_opencv))
    print(f"Diferencia media entre implementaciones: {difference:.6f}")

if __name__ == "__main__":
    main()