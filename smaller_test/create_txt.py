import os
import random

# Path to the folder containing the images
images_folder_path = './smaller_test/'

# Name of the text file
txt_file_name = './allweather.txt'

gt_folder_path = os.path.join(images_folder_path, 'input')
image_files = os.listdir(gt_folder_path)
valid_extensions = ('.jpg', '.png', '.jpeg')
image_files = [f for f in image_files if f.lower().endswith(valid_extensions)]

with open(txt_file_name, 'w') as txt_file:
    for image_name in image_files:
        txt_file.write(os.path.join('./smaller_test/input/', image_name) + '\n')  

print(f'The file "{txt_file_name}" has been created')

