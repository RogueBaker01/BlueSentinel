import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd  # Para cargar el archivo Excel con metadatos
from sklearn.model_selection import train_test_split  # Para dividir datos en entrenamiento y validación
import folium
import cv2
import os

# Configuración de parámetros
data_dir = 'Datasets/'  # Directorio principal de datasets
img_height, img_width = 224, 224
batch_size = 32

# Función para crear un DataFrame de ejemplo con metadatos de imágenes
def crear_metadata_ejemplo():
    """
    Crea un DataFrame de ejemplo que muestra la estructura esperada del archivo info.xlsx
    En un proyecto real, este DataFrame se cargaría directamente desde el archivo Excel
    """
    # Lista de ejemplo con imágenes de diferentes datasets
    imagenes_ejemplo = []
    
    # Obtener algunas imágenes del dataset ADTPA443 como ejemplo
    adtpa443_path = os.path.join(data_dir, 'ADTPA443')
    if os.path.exists(adtpa443_path):
        try:
            archivos = os.listdir(adtpa443_path)[:10]  # Solo primeras 10 como ejemplo
            for archivo in archivos:
                if archivo.endswith('.png'):
                    imagenes_ejemplo.append({
                        'nombre_imagen': f'ADTPA443/{archivo}',
                        'latitud': np.random.uniform(20, 30),  # Coordenadas de ejemplo
                        'longitud': np.random.uniform(-90, -80),
                        'presencia_tiburon': np.random.choice([0, 1])  # Etiqueta aleatoria de ejemplo
                    })
        except Exception as e:
            print(f"Error al acceder a {adtpa443_path}: {e}")
    else:
        print(f"Directorio {adtpa443_path} no encontrado.")
        # Crear datos completamente ficticios si no hay directorio
        for i in range(5):
            imagenes_ejemplo.append({
                'nombre_imagen': f'ejemplo_imagen_{i}.png',
                'latitud': np.random.uniform(20, 30),
                'longitud': np.random.uniform(-90, -80),
                'presencia_tiburon': np.random.choice([0, 1])
            })
        print("Se crearon datos de ejemplo ficticios.")
    
    if len(imagenes_ejemplo) == 0:
        print("ADVERTENCIA: No se pudieron crear datos de ejemplo.")
    
    return pd.DataFrame(imagenes_ejemplo)

# Cargar metadatos desde Excel (o crear ejemplo si no existe)
def cargar_metadatos(archivo_excel='info_imagenes_ejemplo.xlsx'):
    """
    Carga metadatos desde un archivo Excel.
    Si el archivo no existe, crea un DataFrame de ejemplo.
    
    El archivo Excel debe contener las columnas:
    - nombre_imagen: nombre del archivo (incluyendo subcarpeta)
    - latitud: coordenada de latitud
    - longitud: coordenada de longitud  
    - presencia_tiburon: 1 para presencia, 0 para ausencia
    """
    try:
        # Intentar cargar el archivo Excel real
        df = pd.read_excel(archivo_excel)
        print(f"Metadatos cargados desde {archivo_excel}")
        return df
    except FileNotFoundError:
        print(f"Archivo {archivo_excel} no encontrado. Intentando con Info.xlsx...")
        try:
            df = pd.read_excel('Info.xlsx')
            print("Metadatos cargados desde Info.xlsx")
            return df
        except FileNotFoundError:
            print("Ningún archivo de metadatos encontrado. Creando datos de ejemplo...")
            return crear_metadata_ejemplo()
    except Exception as e:
        print(f"Error al cargar {archivo_excel}: {e}")
        print("Creando datos de ejemplo...")
        return crear_metadata_ejemplo()

# Cargar los metadatos
df_metadatos = cargar_metadatos()

print(f"Total de imágenes en metadatos: {len(df_metadatos)}")
print(f"Distribución de etiquetas:\n{df_metadatos['presencia_tiburon'].value_counts()}")

# Función para cargar y preprocesar imágenes desde metadatos
def cargar_imagenes_desde_metadatos(df_metadatos, data_dir):
    """
    Carga y preprocesa imágenes basándose en los metadatos del DataFrame.
    
    Args:
        df_metadatos: DataFrame con metadatos de las imágenes
        data_dir: directorio base donde se encuentran las imágenes
    
    Returns:
        imagenes_procesadas: lista de imágenes normalizadas
        etiquetas: lista de etiquetas correspondientes
    """
    imagenes_procesadas = []
    etiquetas = []
    imagenes_no_encontradas = []
    
    print("Procesando imágenes...")
    
    for idx, fila in df_metadatos.iterrows():
        # Construir la ruta completa a la imagen
        ruta_imagen = os.path.join(data_dir, fila['nombre_imagen'])
        
        try:
            # Cargar la imagen usando OpenCV
            imagen = cv2.imread(ruta_imagen)
            
            if imagen is None:
                imagenes_no_encontradas.append(ruta_imagen)
                continue
            
            # Convertir de BGR a RGB (OpenCV usa BGR por defecto)
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            
            # Redimensionar la imagen al tamaño requerido por el modelo
            imagen_redimensionada = cv2.resize(imagen, (img_width, img_height))
            
            # Normalizar los valores de píxeles al rango [0, 1]
            imagen_normalizada = imagen_redimensionada.astype(np.float32) / 255.0
            
            # Añadir la imagen procesada y su etiqueta a las listas
            imagenes_procesadas.append(imagen_normalizada)
            etiquetas.append(fila['presencia_tiburon'])
            
        except Exception as e:
            print(f"Error procesando imagen {ruta_imagen}: {e}")
            imagenes_no_encontradas.append(ruta_imagen)
    
    if imagenes_no_encontradas:
        print(f"Advertencia: {len(imagenes_no_encontradas)} imágenes no se pudieron cargar.")
        print("Primeras 5 imágenes no encontradas:", imagenes_no_encontradas[:5])
    
    print(f"Imágenes procesadas exitosamente: {len(imagenes_procesadas)}")
    
    return imagenes_procesadas, etiquetas

# Cargar y preprocesar todas las imágenes
imagenes_procesadas, etiquetas = cargar_imagenes_desde_metadatos(df_metadatos, data_dir)

# Validar que se han cargado imágenes correctamente
if len(imagenes_procesadas) == 0:
    print("ERROR: No se pudieron cargar imágenes. Verifique las rutas y archivos.")
    exit(1)

if len(imagenes_procesadas) < 4:
    print("ADVERTENCIA: Se cargaron muy pocas imágenes para entrenar un modelo efectivo.")
    print(f"Se recomienda tener al menos 100 imágenes, pero solo se encontraron {len(imagenes_procesadas)}")

# Convertir las listas a arrays de NumPy para el entrenamiento
X = np.array(imagenes_procesadas)
y = np.array(etiquetas)

print(f"Forma del array de imágenes: {X.shape}")
print(f"Forma del array de etiquetas: {y.shape}")

# Validar que hay suficientes datos para entrenar y validar
if len(X) < 2:
    print("ERROR: Se necesitan al menos 2 imágenes para dividir en entrenamiento y validación.")
    exit(1)

# Dividir los datos en conjuntos de entrenamiento y validación
# Ajustar el test_size para datasets pequeños
test_size = 0.2 if len(X) >= 10 else 0.1 if len(X) >= 5 else 1/len(X)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=test_size,  # Ajustado dinámicamente
    random_state=42,  # Para reproducibilidad
    stratify=y if len(np.unique(y)) > 1 else None  # Solo estratificar si hay más de una clase
)

print(f"Datos de entrenamiento: {X_train.shape[0]} imágenes")
print(f"Datos de validación: {X_val.shape[0]} imágenes")

# Definir la arquitectura del modelo
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Congelar las capas del modelo base para transfer learning
base_model.trainable = False

# Añadir capas personalizadas para clasificación binaria
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # Sigmoid para clasificación binaria

# Crear el modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Resumen del modelo:")
print(f"Total de parámetros: {model.count_params():,}")

# Entrenar el modelo usando los arrays de NumPy en lugar de generadores
print("Iniciando entrenamiento del modelo...")

# Ajustar el batch_size para datasets pequeños
batch_size_actual = min(batch_size, len(X_train))

try:
    history = model.fit(
        X_train, y_train,  # Datos de entrenamiento como arrays de NumPy
        epochs=10,
        batch_size=batch_size_actual,
        validation_data=(X_val, y_val),  # Datos de validación como arrays de NumPy
        verbose=1
    )
    print("Entrenamiento completado exitosamente.")
except Exception as e:
    print(f"Error durante el entrenamiento: {e}")
    print("Intentando con configuración reducida...")
    try:
        # Entrenar con menos epochs y batch size más pequeño
        history = model.fit(
            X_train, y_train,
            epochs=3,
            batch_size=1,
            validation_data=(X_val, y_val),
            verbose=1
        )
        print("Entrenamiento completado con configuración reducida.")
    except Exception as e2:
        print(f"Error crítico en el entrenamiento: {e2}")
        exit(1)

# Guardar el modelo entrenado
try:
    model.save('shark_predictor_model.h5')
    print("Modelo guardado como 'shark_predictor_model.h5'")
except Exception as e:
    print(f"Error al guardar el modelo: {e}")
    print("Intentando guardar en formato alternativo...")
    try:
        model.save('shark_predictor_model', save_format='tf')
        print("Modelo guardado como 'shark_predictor_model' (formato TensorFlow)")
    except Exception as e2:
        print(f"Error crítico al guardar modelo: {e2}")

# Crear visualización en mapa usando las coordenadas del DataFrame
def crear_mapa_predicciones(df_metadatos, model, data_dir, threshold=0.8):
    """
    Crea un mapa interactivo con predicciones del modelo usando las coordenadas del DataFrame.
    
    Args:
        df_metadatos: DataFrame con metadatos de las imágenes
        model: modelo entrenado para hacer predicciones
        data_dir: directorio base de las imágenes
        threshold: umbral de probabilidad para mostrar predicciones (por defecto 0.8)
    """
    # Crear mapa centrado en una ubicación promedio
    lat_promedio = df_metadatos['latitud'].mean()
    lon_promedio = df_metadatos['longitud'].mean()
    
    mapa_predicciones = folium.Map(
        location=[lat_promedio, lon_promedio], 
        zoom_start=6
    )
    
    print("Generando predicciones para el mapa...")
    
    for idx, fila in df_metadatos.iterrows():
        ruta_imagen = os.path.join(data_dir, fila['nombre_imagen'])
        
        try:
            # Cargar y preprocesar la imagen para predicción
            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                continue
                
            # Convertir de BGR a RGB
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            
            # Redimensionar y normalizar
            imagen_redimensionada = cv2.resize(imagen, (img_width, img_height))
            imagen_normalizada = imagen_redimensionada.astype(np.float32) / 255.0
            
            # Añadir dimensión del batch
            imagen_batch = np.expand_dims(imagen_normalizada, axis=0)
            
            # Hacer predicción
            probabilidad = model.predict(imagen_batch, verbose=0)[0][0]
            
            # Determinar color del marcador basado en la probabilidad
            if probabilidad > threshold:
                color = 'red'
                fill_color = 'darkred'
                popup_text = f"Alta probabilidad de tiburón: {probabilidad:.2%}"
            elif probabilidad > 0.5:
                color = 'orange' 
                fill_color = 'orange'
                popup_text = f"Probabilidad media de tiburón: {probabilidad:.2%}"
            else:
                color = 'green'
                fill_color = 'lightgreen'
                popup_text = f"Baja probabilidad de tiburón: {probabilidad:.2%}"
            
            # Añadir información adicional al popup
            popup_text += f"<br>Imagen: {fila['nombre_imagen']}<br>Etiqueta real: {'Presente' if fila['presencia_tiburon'] == 1 else 'Ausente'}"
            
            # Crear marcador en el mapa
            folium.CircleMarker(
                location=[fila['latitud'], fila['longitud']],
                radius=8,
                popup=popup_text,
                color=color,
                fill=True,
                fill_color=fill_color,
                fillOpacity=0.7
            ).add_to(mapa_predicciones)
            
        except Exception as e:
            print(f"Error procesando imagen para mapa {ruta_imagen}: {e}")
    
    return mapa_predicciones

# Crear y guardar el mapa con predicciones
print("Creando mapa interactivo con predicciones...")
try:
    if len(df_metadatos) > 0:
        mapa = crear_mapa_predicciones(df_metadatos, model, data_dir)
        mapa.save("mapa_predicciones.html")
        print("Mapa guardado como 'mapa_predicciones.html'")
    else:
        print("No hay datos suficientes para crear el mapa.")
except Exception as e:
    print(f"Error al crear el mapa: {e}")

# Mostrar estadísticas finales
print("\n=== RESUMEN FINAL ===")
print(f"Total de imágenes procesadas: {len(df_metadatos)}")
print(f"Arquitectura del modelo: MobileNetV2 + capas personalizadas")
print(f"Datos de entrenamiento: {X_train.shape[0]} imágenes")
print(f"Datos de validación: {X_val.shape[0]} imágenes")
print("Archivos generados:")
print("- shark_predictor_model.h5 (modelo entrenado)")

print("- mapa_predicciones.html (mapa interactivo)")
