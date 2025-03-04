import cv2
import numpy as np

def check_separability_property(img, sigma):
    #1D y 1D
    filtered_x = cv2.GaussianBlur(img, (0, 1), sigmaX=sigma, sigmaY=0)
    filtered_xy = cv2.GaussianBlur(filtered_x, (1,0), sigmaX=0, sigmaY=sigma)
    #2D
    filtered_combined = cv2.GaussianBlur(img, (0, 0), sigma)
    
    return filtered_xy, filtered_combined

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        sigma = 3

        filtered_xy, filtered_combined = check_separability_property(frame, sigma)

        cv2.imshow('Original', frame)
        cv2.imshow('Separados', filtered_xy)
        cv2.imshow('Combinados', filtered_combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()