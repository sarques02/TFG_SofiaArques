import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from val_data_functions import ValData
from utils import *
import os
import numpy as np
import random
from Student_transweather_model import Student_Transweather
from transweather_model import Transweather
from datetime import datetime
from collections import OrderedDict
from tqdm import tqdm
import torch.nn.functional as F
from math import log10


parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('--dataset', default='../smaller_test', required=False, help='facades')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--model', type=int, default=0, help='0 original, 1 KD, 2 Light_model')
parser.add_argument('-seed', help='set random seed', default=42, type=int)
opt = parser.parse_args()
print(opt)

val_batch_size = opt.test_batch_size

date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Path al fichero de texto dentro de smaller_test
val_filename = '../../tfg_sofia/smaller_test/allweather.txt'

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)

# --- Load Networks --- #
if opt.model == 0:  # Original
    model_path = 'best_model'
    save_path = './resultados/Original/'
    fichero = f'resultados_original_{date}.txt'
    net = Transweather()
elif opt.model == 1:  # KD
    model_path = 'Best_KD'
    save_path = './resultados/KD/'
    net = Student_Transweather()
    fichero = f'resultados_KD_{date}.txt'
elif opt.model == 2:
    model_path = 'Best_LightModel'
    save_path = './resultados/Light/'
    net = Student_Transweather()
    fichero = f'resultados_Light_{date}.txt'   
elif opt.model == 3:
    model_path = 'Best_Encoder.pth'
    save_path = './resultados/Encoder/'
    net = Student_Transweather()
    fichero = f'resultados_Encoder_{date}.txt'
else:
    model_path = 'Best_Decoder.pth'
    save_path = './resultados/Decoder/'
    net = Student_Transweather()
    fichero = f'resultados_Decoder_{date}.txt' 
net = nn.DataParallel(net, device_ids=device_ids)
net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
net.eval()   

if not os.path.exists(save_path):
    os.makedirs(save_path)

# set seed
seed = opt.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print('Seed:\t{}'.format(seed))

#Para el loader, si las carpetas tienen la estructura de smaller_test, dejar esto así
val_data_dir = ''

val_data_loader = DataLoader(ValData(val_data_dir, val_filename), batch_size=val_batch_size, shuffle=False, num_workers=8)

print(f"Guardando imágenes en {save_path}")


def save_test_image(dehaze, image_name):
    dehaze_images = torch.split(dehaze, 1, dim=0)
    batch_num = len(dehaze_images)
    for ind in range(batch_num):
        utils.save_image(dehaze_images[ind], '{}/{}'.format(save_path, os.path.basename(image_name[ind])[:-3] + 'png'))

def calculate_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return mse_list, psnr_list

print('--- Testing starts! ---')

psnr_list = []
ssim_list = []

t = []
ssims = []
psnrs = []
mses = []

rain_mse_list = []
rain_psnr_list = []
rain_ssim_list = []
haze_mse_list = []
haze_psnr_list = []
haze_ssim_list = []
snow_mse_list = []
snow_psnr_list = []
snow_ssim_list = []

for batch_id, val_data in tqdm(enumerate(val_data_loader)):

    with torch.no_grad():
        haze, gt, image_name = val_data
        haze = haze.to(device)
        gt = gt.to(device)
        start_time = time.time()
        dehaze, _, _ = net(haze) #devuelve clean, encoder, decoder
        dehaze=dehaze.to(device)
        end_time = time.time() - start_time
        t.append(end_time)
        save_test_image(dehaze, image_name)
        mse, psnr = calculate_psnr(dehaze, gt)
        ssim = to_ssim_skimage(dehaze, gt)

        ssims.append(ssim)
        psnrs.append(psnr)
        mses.append(mse)

        if "rain" in str(image_name):
            rain_mse_list.append(mse)
            rain_psnr_list.append(psnr)
            rain_ssim_list.append(ssim)

        if "snow" in str(image_name):
            snow_mse_list.append(mse)
            snow_psnr_list.append(psnr)
            snow_ssim_list.append(ssim)

        if "haze" in str(image_name):
            haze_mse_list.append(mse)
            haze_psnr_list.append(psnr)
            haze_ssim_list.append(ssim)

print('validation time is {0:.4f}'.format(end_time))

# Guarda resultados en el fichero
with open(fichero, "w") as results_file:
    results_file.write(f"Resultados finales:\n")
    results_file.write(f"\nTotal time: {np.sum(t)}\n")
    results_file.write(f"Avg Time per image: {np.mean(t)}\n")
    results_file.write(f"Avg MSE: {np.mean(mses)}\n")
    results_file.write(f"Avg PSNR: {np.mean(psnrs)}\n")
    results_file.write(f"Avg SSIM: {np.mean(ssims)}\n")

    results_file.write("\n Imágenes de lluvia \n")
    results_file.write(f"MSE medio lluvia: {np.mean(rain_mse_list)}\n")
    results_file.write(f"PSNR medio lluvia: {np.mean(rain_psnr_list)}\n")
    results_file.write(f"SSIM medio lluvia: {np.mean(rain_ssim_list)}\n")

    results_file.write("\n Imágenes de nieve \n")
    results_file.write(f"MSE medio nieve: {np.mean(snow_mse_list)}\n")
    results_file.write(f"PSNR medio nieve: {np.mean(snow_psnr_list)}\n")
    results_file.write(f"SSIM medio nieve: {np.mean(snow_ssim_list)}\n")

    results_file.write("\n Imágenes de niebla \n")
    results_file.write(f"MSE medio niebla: {np.mean(haze_mse_list)}\n")
    results_file.write(f"PSNR medio niebla: {np.mean(haze_psnr_list)}\n")
    results_file.write(f"SSIM medio niebla: {np.mean(haze_ssim_list)}\n")