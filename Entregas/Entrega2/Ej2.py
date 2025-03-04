import cv2
import numpy as np

def apply_gaussian_filter(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma)

def check_cascading_property(img, sigma1, sigma2):
    filtered1 = apply_gaussian_filter(img, sigma1)
    #le aplico otra vez el filtro gaussiano a la imagen ya filtrada con otro sigma
    filtered2 = apply_gaussian_filter(filtered1, sigma2)
    
    combined_sigma = np.sqrt(sigma1**2 + sigma2**2)
    filtered_combined = apply_gaussian_filter(img, combined_sigma)
    
    return filtered2, filtered_combined

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        sigma1 = 2
        sigma2 = 3

        filtered2, filtered_combined = check_cascading_property(frame, sigma1, sigma2)

        # cv2.imshow('Original', frame)
        # cv2.imshow('2 Filtros', filtered2)
        # cv2.imshow('Combinados', filtered_combined)

        #puedo hacer un hstack para mostrar las imagenes juntas
        # cv2.imshow('Comparacion', np.hstack([frame, filtered2, filtered_combined]))

        #ahora cambio el tamaño de la imagen para que se vea mejor y junto las imagenes filtradas con la original
        cv2.imshow('Comparacion', cv2.resize(np.hstack([frame, filtered2, filtered_combined]), (0,0), fx=0.75, fy=0.75))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()