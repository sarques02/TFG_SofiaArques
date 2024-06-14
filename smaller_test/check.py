import os

def get_image_names(folder_path):
    # Obtener la lista de nombres de archivos en la carpeta
    return set(os.listdir(folder_path))

def delete_images(folder, images):
    for image in images:
        image_path = os.path.join(folder, image)
        if os.path.isfile(image_path):
            os.remove(image_path)
            print(f"Eliminado: {image_path}")

def compare_and_delete_images(folder1, folder2):
    # Obtener nombres de imágenes en ambas carpetas
    images_folder1 = get_image_names(folder1)
    images_folder2 = get_image_names(folder2)
    
    # Encontrar las diferencias
    only_in_folder1 = images_folder1 - images_folder2
    only_in_folder2 = images_folder2 - images_folder1
    
    # Eliminar imágenes solo en la primera carpeta
    print("Eliminando imágenes solo en", folder1)
    delete_images(folder1, only_in_folder1)
    
    # Eliminar imágenes solo en la segunda carpeta
    print("Eliminando imágenes solo en", folder2)
    delete_images(folder2, only_in_folder2)

# Ejemplo de uso
folder1_path = 'input'
folder2_path = 'gt'
compare_and_delete_images(folder1_path, folder2_path)
