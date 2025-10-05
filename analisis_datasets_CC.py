import os
import numpy as np
import cv2

input_path = 'C:/Users/Rogue/OneDrive/Documents/BlueSentinel/ADTPA443'
output_path = 'C:/Users/Rogue/OneDrive/Documents/BlueSentinel/ADTPA443_Results'

def analisis_clorofila(img_path):
    """
    Analiza una imagen para resaltar las áreas de alta concentración de clorofila
    (colores amarillos, naranjas y rojos).
    """
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: No se pudo leer la imagen en la ruta: {img_path}")
        return None, None
    
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
    lower_yellow_orange = np.array([20, 100, 100])
    upper_yellow_orange = np.array([40, 255, 255])
    mask_yellow_orange = cv2.inRange(hsv_img, lower_yellow_orange, upper_yellow_orange)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask_red2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    final_mask = cv2.bitwise_or(mask_yellow_orange, mask_red)
    
    result = cv2.bitwise_and(img, img, mask=final_mask)
    
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    return result_rgb, image_rgb

if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"Carpeta de salida creada en: {output_path}")

try:
    filenames = os.listdir(input_path)
except FileNotFoundError:
    print(f"Error: La carpeta de entrada no fue encontrada en la ruta: {input_path}")
    filenames = []

print("Iniciando análisis de imágenes...")

for filename in filenames:
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        full_image_path = os.path.join(input_path, filename)
        print(f"Procesando imagen: {filename}...")

        result_rgb, img_rgb = analisis_clorofila(full_image_path)

        if result_rgb is not None:
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_clorofila{ext}"
            final_output_path = os.path.join(output_path, new_filename)
            
            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(final_output_path, result_bgr)
            
            print(f"  -> ¡Resultado guardado en: {final_output_path}!")

print("\nProceso finalizado.")