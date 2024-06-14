import os
import random

# Ruta a la carpeta que contiene las imágenes
ruta_carpeta_imagenes = '/home/gatv-matteo/Desktop/tfg_sofia/TransWeather_Quantization/data/smaller_test/'

# Nombre del archivo de texto
nombre_archivo_txt = './allweather.txt'

# Obtener nombres de archivos de la carpeta de imágenes (en la carpeta gt)
ruta_carpeta_gt = os.path.join(ruta_carpeta_imagenes, 'input')
archivos_imagenes = os.listdir(ruta_carpeta_gt)

# Filtrar por extensiones de imagen válidas (ajusta según tus extensiones)
extensiones_validas = ('.jpg', '.png', '.jpeg')
archivos_imagenes = [f for f in archivos_imagenes if f.lower().endswith(extensiones_validas)]

# Seleccionar aleatoriamente 50 imágenes
# archivos_aleatorios = random.sample(archivos_imagenes, 50)

# Crear o sobrescribir el archivo de texto con los nombres de archivos
with open(nombre_archivo_txt, 'w') as archivo_txt:
    for nombre_imagen in archivos_imagenes:
        archivo_txt.write(os.path.join('/data/smaller_test/input/', nombre_imagen) + '\n')  # Agregar etiqueta 0

print(f'Se ha creado el archivo "{nombre_archivo_txt}"')

# Luego puedes usar este archivo en la función proporcionada
# Si image_list es None, se utilizará el archivo recién creado
