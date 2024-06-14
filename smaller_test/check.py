import os
import argparse

def get_image_names(folder_path):
    # Get the list of file names in the folder
    return set(os.listdir(folder_path))

def delete_images(folder, images):
    for image in images:
        image_path = os.path.join(folder, image)
        if os.path.isfile(image_path):
            os.remove(image_path)
            print(f"Deleted: {image_path}")

def compare_images(folder1, folder2, delete=False):
    # Get image names in both folders
    images_folder1 = get_image_names(folder1)
    images_folder2 = get_image_names(folder2)
    
    # Find the differences
    only_in_folder1 = images_folder1 - images_folder2
    only_in_folder2 = images_folder2 - images_folder1
    
    # Handle images only in the first folder
    if delete:
        print("Deleting images only in", folder1)
        delete_images(folder1, only_in_folder1)
    else:
        print("Images only in", folder1)
        for image in only_in_folder1:
            print(image)
    
    # Handle images only in the second folder
    if delete:
        print("Deleting images only in", folder2)
        delete_images(folder2, only_in_folder2)
    else:
        print("Images only in", folder2)
        for image in only_in_folder2:
            print(image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare images in two folders and optionally delete the different ones")
    parser.add_argument("folder1", type=str, help="Path to the first folder")
    parser.add_argument("folder2", type=str, help="Path to the second folder")
    parser.add_argument("--delete", action="store_true", help="Delete different images if specified")
    
    args = parser.parse_args()
    
    compare_images(args.folder1, args.folder2, args.delete)
