import cv2
import numpy as np
import time

def manual_convolution(image, kernel):
    
    img_h, img_w = image.shape
    kernel_h, kernel_w = kernel.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2
    
    # Agregar padding a la imagen
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros_like(image)
    
    # Aplicar la convolución
    for i in range(img_h):
        for j in range(img_w):
            region = padded_image[i:i+kernel_h, j:j+kernel_w]
            output[i, j] = np.sum(region * kernel)
    
    return output


def main():
    imageOriginal=cv2.imread('img.png')
    image = cv2.cvtColor(imageOriginal, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), dtype=np.float32) / 9
    
    

    start_time = time.time()
    manual_result = manual_convolution(image, kernel)
    manual_time = time.time() - start_time

    start_time = time.time()
    opencv_result = cv2.filter2D(image, -1, kernel)
    opencv_time = time.time() - start_time

    cv2.imshow('Original', imageOriginal)
    cv2.imshow('Convolucion Manual', manual_result)
    cv2.imshow('Convolucion OpenCV', opencv_result)

    print(f"Manual convolution time: {manual_time:.6f} seconds")
    print(f"OpenCV convolution time: {opencv_time:.6f} seconds")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()