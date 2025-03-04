import cv2
import numpy as np
from umucv.util import ROI, putText
from umucv.stream import autoStream

current_filter = 'none'
sigma = 1
sigmacol = 1
rad = 1
low_threshold = 0
high_threshold = 0
mostrado=False
crome=False
soloRoi=False
cv2.namedWindow('Filtered Video')
region = ROI("Filtered Video")


def draw_help_menu(img):
    overlay = img.copy()
    x, y = 20, 40  # Posición inicial del texto
    spacing = 30   # Espaciado entre líneas
    
    # Fondo semitransparente para el menú
    cv2.rectangle(overlay, (0, 0), (500, 500), (50, 50, 50), -1)
    alpha = 0.7
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    # Texto del menú
    text_lines = [
        "BLUR FILTERS",
        "",
        "0: do nothing",
        "1: box",
        "2: Gaussian",
        "3: median",
        "4: bilateral",
        "5: min",
        "6: max",
        "7: canny",
        "",
        "c: color/monochrome",
        "r: only roi",
        "",
        "h: show/hide help"
    ]
    
    for i, line in enumerate(text_lines):
        color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7 if i == 0 else 0.6  # Título más grande
        thickness = 2 if i == 0 else 1
        cv2.putText(img, line, (x, y + i * spacing), font, scale, color, thickness, cv2.LINE_AA)
    return img

def update_sigma(val):
    global sigma
    sigma = val

def update_sigmacol(val):
    global sigmacol
    sigmacol = val

def update_rad(val):
    global rad
    rad = val

def update_low(val):
    global low_threshold
    low_threshold = val

def update_high(val):
    global high_threshold
    high_threshold = val

def box_filter_integral(img, ksize):
    """
    Aplica el Box Filter usando la Imagen Integral en imágenes en color.
    
    :param img: Imagen de entrada (BGR o escala de grises)
    :param ksize: Tamaño del kernel (ksize x ksize)
    :return: Imagen filtrada
    """
    # Si la imagen es en escala de grises, conviértela a 3 canales
    if len(img.shape) == 3:

        # Separar los canales B, G y R
        b, g, r = cv2.split(img)

        # Aplicar el Box Filter a cada canal
        b_filtered = apply_box_filter(b, ksize)
        g_filtered = apply_box_filter(g, ksize)
        r_filtered = apply_box_filter(r, ksize)

        # Combinar los canales filtrados
        filtered_img = cv2.merge([b_filtered, g_filtered, r_filtered])
    else:
        filtered_img = apply_box_filter(img, ksize)
    return filtered_img

def apply_box_filter(img, ksize):
    """
    Aplica el Box Filter usando la Imagen Integral.
    
    :param img: Imagen de entrada en escala de grises
    :param ksize: Tamaño del kernel (ksize x ksize)
    :return: Imagen filtrada
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)

    integral_img = cv2.integral(img)

    h, w = img.shape
    output = np.zeros((h, w), dtype=np.float32)
    
    offset = ksize // 2 #para centrar la ventana ya que cv2.integral() añade una fila y columna de 0s
    
    for y in range(h):
        for x in range(w):
            x1, y1 = max(x - offset, 0), max(y - offset, 0)
            x2, y2 = min(x + offset, w - 1), min(y + offset, h - 1)

            S = (integral_img[y2+1, x2+1] - integral_img[y1, x2+1]
                 - integral_img[y2+1, x1] + integral_img[y1, x1])

            num_pixels = (y2 - y1 + 1) * (x2 - x1 + 1)
            #para cada pixel calculo la media de los pixeles de la ventana
            output[y, x] = S / num_pixels

    return np.uint8(output)  

def apply_filter(frame, filter_type, sigma, sigmacol, rad, low, high):
    if filter_type == 'box':
        return box_filter_integral(frame, sigma*2+1)
    elif filter_type == 'gaussian':
        return cv2.GaussianBlur(frame, (sigma*2+1, sigma*2+1), 0)
    elif filter_type == 'median':
        return cv2.medianBlur(frame, sigma*2+1)
    elif filter_type == 'bilateral':
        return cv2.bilateralFilter(frame, rad, sigmacol*2, sigma*2)
    elif filter_type == 'min':
        return cv2.erode(frame, np.ones((sigma, sigma), np.uint8))
    elif filter_type == 'max':
        return cv2.dilate(frame, np.ones((sigma, sigma), np.uint8))
    elif filter_type == 'canny':
        return cv2.Canny(frame, low, high, 3)
    else:
        return frame

def handle_key_press(key):
    global current_filter
    if key == ord('0'):
        current_filter = 'none'
    elif key == ord('1'):
        current_filter = 'box'
    elif key == ord('2'):
        current_filter = 'gaussian'
    elif key == ord('3'):
        current_filter = 'median'
    elif key == ord('4'):
        current_filter = 'bilateral'
    elif key == ord('5'):
        current_filter = 'min'
    elif key == ord('6'):
        current_filter = 'max'
    elif key == ord('7'):
        current_filter = 'canny'

# Main function
def main():
    global current_filter, sigma, sigmacol, rad, mostrado, crome, soloRoi

    cap = cv2.VideoCapture(0)

    roi=None
    cv2.createTrackbar('Sigma', 'Filtered Video', 1, 500, update_sigma)
    cv2.createTrackbar('SigmaCol', 'Filtered Video', 1, 500, update_sigmacol)
    cv2.createTrackbar('Rad', 'Filtered Video', 1, 50, update_rad)
    cv2.createTrackbar('Low Threshold', 'Filtered Video', 0, 255, update_low)
    cv2.createTrackbar('High Threshold', 'Filtered Video', 0, 255, update_high)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF
        handle_key_press(key)

        if key == ord('q'):
            break
        elif key == ord('r'):
            if soloRoi:
                soloRoi=False
            else:
                soloRoi=True
        elif key == ord('c'):
            if crome:
                crome=False
            else:
                crome=True
        elif key == ord('h'):
            if(mostrado):
                cv2.destroyWindow("Help Menu")
                mostrado=False
            elif(not mostrado):
                mostrado=True
                # Mostrar una ventana con el menu de ayuda
                
                img = np.zeros((500, 500, 3), dtype=np.uint8)
                menu_img = draw_help_menu(img)

                cv2.imshow("Help Menu", menu_img)
        if crome:
            if len(frame.shape) == 3:
                frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            if len(frame.shape) == 2:
                frame=cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if soloRoi:
            if region.roi:
                [x1,y1,x2,y2] = region.roi
                roi = frame[y1:y2+1, x1:x2+1]
                if key == ord('x'):
                    region.roi = []
                # cv2.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=1)
            if roi is None:
                roi = frame[100:300, 100:300]
                [x1,y1,x2,y2] = [100,100,299,299]
        else:
            roi = frame
        
        
        filtered_roi = apply_filter(roi, current_filter, sigma, sigmacol, rad, low_threshold, high_threshold)

        if len(filtered_roi.shape) == 2 and len(frame.shape)==3:
            filtered_roi = cv2.cvtColor(filtered_roi, cv2.COLOR_GRAY2BGR)

        if(roi.size == frame.size):
            frame = filtered_roi
        else:
            frame[y1:y2+1, x1:x2+1] = filtered_roi

        cv2.imshow('Filtered Video', frame)

        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()