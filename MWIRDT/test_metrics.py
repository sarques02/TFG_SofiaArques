import argparse
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import is_image_file, load_img, save_img
torch.backends.cudnn.benchmark = True
import time
from metrics import *
from dataset_test import DatasetFromFolder_Test
from os.path import join
from datetime import datetime

import network_model1 as network_haze1
import network_model2 as network_rain2
import network_model3 as network_snow3


#Poner fecha en ficheros
#Poner detección de dispositivo


# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', default='../smaller_test', required=False, help='facades')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--factor', default= 4, type=int)
parser.add_argument('--models', type=int, default=0, help='0 original, 1 KD, 2 Light_model')

opt = parser.parse_args()
print(opt)
factor_value = int(opt.factor)


# --- Device --- #
# device_ids = [Id for Id in range(torch.cuda.device_count())] #Si hay varias gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if opt.models == 0:
    print("Usando modelos originales")
    d1_path = f"./checkpoints_originales/domain1.pth" #Path a los pesos
    d2_path = f"./checkpoints_originales/domain2.pth"
    d3_path = f"./checkpoints_originales/domain3.pth"
    save_img_path = f"results/Original_net" #Carpeta donde se guardan las imágenes
    file = f"resultados_original_{date}.txt" #fichero donde se apuntan los resultados
    factor_value = 1 #Factor = 1 para las redes originales
    
elif opt.models == 1:
    print("Usando modelos KD")
    d1_path = f"./KD_models/net1factor{factor_value}.pth"
    d2_path = f"./KD_models/net2factor{factor_value}.pth"
    d3_path = f"./KD_models/net3factor{factor_value}.pth"
    save_img_path = f"results/KD_Model/factor{factor_value}" 
    file = f"resultados_KD_{date}.txt"

else:
    print("Usando modelos ligeros")
    d1_path = f"./Light_models/net1factor{factor_value}.pth"
    d2_path = f"./Light_models/net2factor{factor_value}.pth"
    d3_path = f"./Light_models/net3factor{factor_value}.pth"
    save_img_path = f"results/Light_models/factor{factor_value}"
    file = f"resultados_Light_{date}.txt"
    

if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)


## Load Networks ##
G_path = "./checkpoints_originales/netG_model.pth" #Como no se comprime la red de restauración siempre será este path
my_net = torch.load(G_path)
my_net.to(device)
my_net.eval()


d1_net = network_haze1.My_net(factor = factor_value)
#si es el original carga directa, si es kd o student carga state dict
if opt.models == 0:
    d1_net = torch.load(d1_path)
else:
    d1_net.load_state_dict(torch.load(d1_path, map_location=device))
d1_net.to(device)
d1_net.eval()

d2_net = network_rain2.My_net(factor = factor_value)
if opt.models == 0:
    d2_net = torch.load(d2_path)
else:
    d2_net.load_state_dict(torch.load(d2_path, map_location=device))
d2_net.to(device)
d2_net.eval()

d3_net = network_snow3.My_net(factor = factor_value)
if opt.models == 0:
    d3_net = torch.load(d3_path)
else:
    d3_net.load_state_dict(torch.load(d3_path, map_location=device))

d3_net.to(device)
d3_net.eval()

# ---------  Load Datasets  ------------#

def get_test_set(root_dir):
    test_dir = join(root_dir)

    return DatasetFromFolder_Test(test_dir) 

image_dir = "{}/".format(opt.dataset)
image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]
transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform = transforms.Compose(transform_list)

test_set = get_test_set(opt.dataset+"/input")
testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.test_batch_size, shuffle=False)

gt_set=get_test_set(opt.dataset+"/gt")
gt_data_loader = DataLoader(dataset=gt_set, batch_size=opt.test_batch_size, shuffle=False)


t = []
ssims=[]
psnrs=[]
mses=[]

rain_mse_list = []
rain_psnr_list = []
rain_ssim_list = []
haze_mse_list = []
haze_psnr_list = []
haze_ssim_list = []
snow_mse_list = []
snow_psnr_list = []
snow_ssim_list = []

for iteration_test, batch in enumerate(zip(testing_data_loader, gt_data_loader)):
    real_a, filename = batch[0][0].to(device), batch[0][1]
    gt, gt_filename = batch[1][0].to(device), batch[1][1]
    real_a = real_a.to(device)
    gt = gt.to(device)
    gt_path = "gt/"+str(filename)

    t_inicial = time.time() 

    fake_d1 = d1_net(real_a).to(device)
    fake_d2 = d2_net(real_a).to(device)
    fake_d3 = d3_net(real_a).to(device)
    fake_b = my_net(real_a, fake_d1, fake_d2, fake_d3).to(device)
    t_pred= time.time()-t_inicial
    t.append(t_pred)

    out = fake_b 
    out_img = out[0].detach().squeeze(0).to(device)
    gt_image = gt.cpu().numpy().squeeze()

    ssim = calculate_ssim(gt_image, out_img)
    psnr, mse = calculate_psnr(gt_image, out_img)
    print("SSIM: ", ssim) 
    print("PSNR: ", psnr)
    print("MSE: ", mse)

    ssims.append(ssim)
    psnrs.append(psnr)
    mses.append(mse)

    if "rain" in str(filename):
        rain_mse_list.append(mse)
        rain_psnr_list.append(psnr)
        rain_ssim_list.append(ssim)


    if "snow" in str(filename):
        snow_mse_list.append(mse)
        snow_psnr_list.append(psnr)
        snow_ssim_list.append(ssim)

    if "haze" in str(filename):
        haze_mse_list.append(mse)
        haze_psnr_list.append(psnr)
        haze_ssim_list.append(ssim)

    #Guardar imágenes
    save_img(out_img, os.path.join(save_img_path, filename[0]))

#### Save Results ###
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

print('##################### Testing Successfully Completed###################################')    


print('Restored Results are saved in the results folder')