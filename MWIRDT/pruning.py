from __future__ import print_function
import argparse
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import is_image_file, load_img, save_img
torch.backends.cudnn.benchmark = True
import time
from dataset_test import DatasetFromFolder_Test
from os.path import join
from metrics import *
import network_model1 as Student_haze
import network_model2 as Student_rain
import network_model3 as Student_snow
import datetime
import torch.nn.utils.prune as prune


'''
Cambiar ruta a dataset en paser. Usar smaller_test
'''



# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', default='../../tfg_sofia/smaller_test', required=False, help='facades')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--cuda', default= False, action='store_false',  help='use cuda')


opt = parser.parse_args()


def get_test_set(root_dir):
    test_dir = join(root_dir)

    return DatasetFromFolder_Test(test_dir)

if opt.cuda:
    torch.cuda.manual_seed(123)
device = torch.device("cuda:0")
print(device)

date = datetime.datetime.now().strftime('%d-%m-%y-%H_%M')

file = f"resultados_Prunning_{date}.txt" #fichero donde se apuntan los resultados
save_img_path = f"results/Prunning/" #Carpeta donde se guardan las imágenes

if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)	


G_path = "checkpoints_originales/netG_model.pth"
my_net = torch.load(G_path).to(device) 
my_net.eval()

d1_path = "checkpoints_originales/domain1.pth"
d1_net = torch.load(d1_path).to(device)
d1_net.eval()

d2_path = "checkpoints_originales/domain2.pth"
d2_net = torch.load(d2_path).to(device)
d2_net.eval()

d3_path = "checkpoints_originales/domain3.pth"
d3_net = torch.load(d3_path).to(device)
d3_net.eval()


image_dir = "{}/".format(opt.dataset)
image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]
transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform = transforms.Compose(transform_list)

test_set = get_test_set(opt.dataset+"/input")
testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.test_batch_size, shuffle=False)


gt_set=get_test_set(opt.dataset+"/gt")
gt_data_loader = DataLoader(dataset=gt_set, batch_size=opt.test_batch_size, shuffle=False)

### ----- PRUNING ------###
def prune_model(model, amount):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
        elif isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
    return model

my_net_pruned = prune_model(my_net, amount=0.2)
d1_net_pruned = prune_model(d1_net, amount=0.2)
d2_net_pruned = prune_model(d2_net, amount=0.2)
d3_net_pruned = prune_model(d3_net, amount=0.2)

def remove_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            prune.remove(module, 'weight')
    return model

my_net_pruned = remove_pruning(my_net_pruned)
d1_net_pruned = remove_pruning(d1_net_pruned)
d2_net_pruned = remove_pruning(d2_net_pruned)
d3_net_pruned = remove_pruning(d3_net_pruned)

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

inspect_weights(d1_net_pruned)



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
    gt_path = "gt/"+str(filename)

    t_inicial = time.time() 

    fake_d1 = d1_net_pruned(real_a)
    fake_d2 = d2_net_pruned(real_a)
    fake_d3 = d3_net_pruned(real_a)
    fake_b = my_net_pruned(real_a, fake_d1, fake_d2, fake_d3)
    t_pred= time.time()-t_inicial
    t.append(t_pred)

    out = fake_b 
    out_img = out[0].detach().squeeze(0).to('cpu')
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

    #Guardar solo 5 imágenes de cada tipo
    save_img(out_img, os.path.join(save_img_path, filename[0]))

print('##################### Testing Successfully Completed###################################')    
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