import numpy as np
from PIL import Image
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as T

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    if isinstance(image_tensor, torch.Tensor):
        # Move the tensor to the CPU
        image_tensor = image_tensor.cpu().float()
        image_numpy = image_tensor.numpy()
    else:
        image_numpy = image_tensor

    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)

    cv2.imwrite(filename, image_numpy)
    print("Image saved as {}".format(filename),end='\r')
