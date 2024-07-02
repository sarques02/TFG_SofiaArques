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
from transweather_model import Transweather




#Cambiar paths:
#Es el path al fichero de texto dentro de smaller_test
val_filename = '../data/smaller_test/allweather.txt' 
model = 'best_model'
save_path =('./Pruning/')

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str)
parser.add_argument('-seed', help='set random seed', default=42, type=int)
args = parser.parse_args()

val_batch_size = args.val_batch_size
exp_name = args.exp_name


#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

#Dejar esto así
val_data_dir = ''

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


val_data_loader = DataLoader(ValData(val_data_dir,val_filename), batch_size=val_batch_size, shuffle=False, num_workers=8)

print(f"Guardando imágenes en {save_path}")



net = Transweather().cuda()
net = nn.DataParallel(net, device_ids=device_ids)
net.load_state_dict(torch.load(model))
net.eval()

### ---- Pruning ---- ###
import torch.nn.utils.prune as prune
def prune_model(model, amount):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # n = 1 and dim = 1, where n represents l1 norm or l2 norm and dim represents 0 if you want to prune full conv filters or 1 if you want to prune channels 
            prune.l1_unstructured(module, name='weight',amount=amount)
        elif isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight',amount=amount)
    return model

net_pruned = prune_model(net, amount=0.2)

def remove_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            prune.remove(module, 'weight')
    return model

net_pruned = remove_pruning(net_pruned)


def inspect_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight = param.data.cpu().numpy()
            zero_count = (weight == 0).sum()
            total_count = weight.size
            sparsity = 100.0 * zero_count / total_count
            print(f'Layer: {name} | Sparsity: {sparsity:.2f}%')
            # Optional: Print a small sample of the weights
            print(f'Weights sample from {name}: {weight.flatten()[:10]}')

inspect_weights(net_pruned)

if not os.path.exists(save_path):     
    os.makedirs(save_path)

# List of images to save
saved_images = {"17_R_rain.jpg", '25_R_rain.jpg', '24_rain.png', '130_rain.png', '123_rain.png', 'haze_0003_s100_a04.png', 'haze_0005_s100_a04.png', 'haze_0011_s80_a05.png', 'haze_0013_s85_a06.png', 'haze_0014_s80_a04.png', 'snow_00018.jpg', 'snow_00075.jpg', 'snow_00148.jpg', 'snow_00202.jpg', 'snow_00270.jpg'}

# Function to save test images
def save_test_image(dehaze, image_name):
    dehaze_images = torch.split(dehaze, 1, dim=0)
    batch_num = len(dehaze_images)    
    for ind in range(batch_num):
        image_basename = os.path.basename(image_name[ind])
        if image_basename in saved_images:
            utils.save_image(dehaze_images[ind], '{}/{}'.format(save_path, image_basename[:-3] + 'png'))

# Function to calculate PSNR
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
        dehaze = net_pruned(haze)
        end_time = time.time() - start_time

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

        save_test_image(dehaze, image_name)

print('validation time is {0:.4f}'.format(end_time))
import datetime
date = datetime.datetime.now().strftime('%d-%m-%y-%H_%M')

file = f"resultados_Prunning_{date}.txt"

with open(file, "w") as results_file:
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

print('Restored Results are saved in the results folder')