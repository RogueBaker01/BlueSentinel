import os
import pandas as pd
import numpy as np
import cv2
import re

EXCEL_FILE_PATH = 'C:/Users/Rogue/OneDrive/Documents/BlueSentinel/Info.xlsx'

input_path_diurno = 'C:/Users/Rogue/OneDrive/Documents/BlueSentinel/SSTD'
input_path_nocturno = 'C:/Users/Rogue/OneDrive/Documents/BlueSentinel/SSTN'
output_path_diurno = 'C:/Users/Rogue/OneDrive/Documents/BlueSentinel/SSTD_Results'
output_path_nocturno = 'C:/Users/Rogue/OneDrive/Documents/BlueSentinel/SSTN_Results'

COL_NOMBRE = 'NOMBRE'
COL_TEMP = 'TEMPERATURA'
COL_ACTIVIDAD = 'ACTIVIDAD'

MAPA_TEMP_MIN = 4.0
MAPA_TEMP_MAX = 32.0


def parse_temp_string(temp_str):
    try:
        numeros = re.findall(r'[-+]?\d*\.\d+|\d+', temp_str)
        if len(numeros) >= 2:
            return float(numeros[0]), float(numeros[1])
        return None, None
    except:
        return None, None

def map_temp_to_hue(temp, temp_min_mapa, temp_max_mapa):
    normalized_temp = (temp - temp_min_mapa) / (temp_max_mapa - temp_min_mapa)
    hue = 120 - (normalized_temp * 120)
    return np.clip(hue, 0, 179)

def analizar_habitat_por_temperatura(img, hue_min, hue_max):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if hue_min <= hue_max:
        lower_bound = np.array([hue_min, 100, 100])
        upper_bound = np.array([hue_max, 255, 255])
        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    else:
        lower1 = np.array([hue_min, 100, 100])
        upper1 = np.array([179, 255, 255])
        mask1 = cv2.inRange(hsv_img, lower1, upper1)
        lower2 = np.array([0, 100, 100])
        upper2 = np.array([hue_max, 255, 255])
        mask2 = cv2.inRange(hsv_img, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    return cv2.bitwise_and(img, img, mask=mask)

def procesar_imagenes_para_tiburon(tiburon_info, input_path, output_path, tipo_actividad):
    nombre_tiburon, hue_min, hue_max = tiburon_info
    print(f"  -> Analizando set de datos '{tipo_actividad}'...")
    shark_name_safe = re.sub(r'[^a-zA-Z0-9_]', '', nombre_tiburon.replace(' ', '_'))
    species_output_path = os.path.join(output_path, shark_name_safe)
    if not os.path.exists(species_output_path):
        os.makedirs(species_output_path)
    
    for filename in os.listdir(input_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_image_path = os.path.join(input_path, filename)
            img = cv2.imread(full_image_path)
            if img is None: continue

            resultado_img = analizar_habitat_por_temperatura(img, hue_min, hue_max)
            
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_{shark_name_safe}_{tipo_actividad}{ext}"
            
            final_output_path = os.path.join(species_output_path, new_filename)
            cv2.imwrite(final_output_path, resultado_img)


print("Cargando datos de tiburones desde 'Info.xlsx'...")
try:
    df_tiburones = pd.read_excel(EXCEL_FILE_PATH)
    print(f"Se encontraron datos de {len(df_tiburones)} especies.")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo Excel en: {EXCEL_FILE_PATH}")
    df_tiburones = pd.DataFrame()

if not os.path.exists(output_path_diurno): os.makedirs(output_path_diurno)
if not os.path.exists(output_path_nocturno): os.makedirs(output_path_nocturno)

for index, tiburon in df_tiburones.iterrows():
    nombre_tiburon = tiburon[COL_NOMBRE]
    temp_str = str(tiburon[COL_TEMP])
    actividad = str(tiburon.get(COL_ACTIVIDAD, 'Ambos')).strip().capitalize()
    
    temp_min_tiburon, temp_max_tiburon = parse_temp_string(temp_str)
    if temp_min_tiburon is None:
        print(f"Advertencia: No se pudo leer el rango de temperatura para '{nombre_tiburon}'. Saltando.")
        continue

    print(f"\nProcesando para: {nombre_tiburon} (Actividad: {actividad})")
    
    hue_max = map_temp_to_hue(temp_min_tiburon, MAPA_TEMP_MIN, MAPA_TEMP_MAX)
    hue_min = map_temp_to_hue(temp_max_tiburon, MAPA_TEMP_MIN, MAPA_TEMP_MAX)
    
    tiburon_info = (nombre_tiburon, hue_min, hue_max)
    
    if actividad == 'Diurno':
        procesar_imagenes_para_tiburon(tiburon_info, input_path_diurno, output_path_diurno, "Diurno")
    elif actividad == 'Nocturno':
        procesar_imagenes_para_tiburon(tiburon_info, input_path_nocturno, output_path_nocturno, "Nocturno")
    elif actividad == 'Ambos':
        procesar_imagenes_para_tiburon(tiburon_info, input_path_diurno, output_path_diurno, "Diurno")
        procesar_imagenes_para_tiburon(tiburon_info, input_path_nocturno, output_path_nocturno, "Nocturno")
    else:
        print(f"Advertencia: Actividad '{actividad}' no reconocida para '{nombre_tiburon}'. Se procesarán ambos sets de datos.")
        procesar_imagenes_para_tiburon(tiburon_info, input_path_diurno, output_path_diurno, "Diurno")
        procesar_imagenes_para_tiburon(tiburon_info, input_path_nocturno, output_path_nocturno, "Nocturno")

print("\nProceso finalizado.")