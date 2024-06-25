from __future__ import print_function
import argparse
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_test import get_test_set
from utils import is_image_file, load_img, save_img
torch.backends.cudnn.benchmark = True
import time
import ast 
from metrics import *

import network_model1 as Student_haze
import network_model2 as Student_rain
import network_model3 as Student_snow


# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', default='../../TransWeather_Quantization/data/smaller_test', required=False, help='facades')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--cuda', default= False, action='store_false',  help='use cuda')
parser.add_argument('--factor', default= 4, type=int)
parser.add_argument('--KD', type=ast.literal_eval, default=True, help='boolean flag for KD')

opt = parser.parse_args()
print(opt)
factor_value = int(opt.factor)

if opt.cuda:
    torch.cuda.manual_seed(123)
device = torch.device("cuda:0")
print(device)

# file = f"{opt.save_path}/resultados_Prunning.txt" #fichero donde se apuntan los resultados


if opt.KD:
    d1_path = f"../models/net1/factor{factor_value}/BestStudent.pth"
    d2_path = f"../models/net2/factor{factor_value}/BestStudent.pth"
    d3_path = f"../models/net3/factor{factor_value}/BestStudent.pth"
    save_img_path = f"results/KD_Model/factor{factor_value}" #Carpeta donde se guardan las imágenes
else:
    d1_path = f"../OnlyStudent/net1/factor{factor_value}/BestStudent.pth"
    d2_path = f"../OnlyStudent/net2/factor{factor_value}/BestStudent.pth"
    d3_path = f"../OnlyStudent/net3/factor{factor_value}/BestStudent.pth"
    save_img_path = f"results/Only_Student/factor{factor_value}" #Carpeta donde se guardan las imágenes
    

if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)

## Load Networks ##
G_path = "../checkpoints/netG_model.pth"
my_net = torch.load(G_path).to(device) 
my_net.eval()


d1_net = Student_haze.My_net(factor = factor_value)
d1_net.load_state_dict(torch.load(d1_path, map_location=device))
d1_net.to(device)
d1_net.eval()


d2_net = Student_rain.My_net(factor = factor_value)
d2_net.load_state_dict(torch.load(d2_path, map_location=device))
d2_net.to(device)
d2_net.eval()

d3_net = Student_snow.My_net(factor = factor_value)
d3_net.load_state_dict(torch.load(d3_path, map_location=device))
d3_net.to(device)
d3_net.eval()

# ---------------------#

image_dir = "{}/".format(opt.dataset)
image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]
transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform = transforms.Compose(transform_list)

test_set = get_test_set(opt.dataset+"/input")
testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.test_batch_size, shuffle=False)


gt_set=get_test_set(opt.dataset+"/gt")
gt_data_loader = DataLoader(dataset=gt_set, batch_size=opt.test_batch_size, shuffle=False)

saved_images = ["17_R_rain.jpg", '25_R_rain.jpg', '24_rain.png', '130_rain.png', '123_rain.png', 'haze_0003_s100_a04.png', 'haze_0005_s100_a04.png', 'haze_0011_s80_a05.png', 'haze_0013_s85_a06.png', 'haze_0014_s80_a04.png', 'snow_00018.jpg', 'snow_00075.jpg', 'snow_00148.jpg', 'snow_00202.jpg', 'snow_00270.jpg']


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

    fake_d1 = d1_net(real_a)
    fake_d2 = d2_net(real_a)
    fake_d3 = d3_net(real_a)
    fake_b = my_net(real_a, fake_d1, fake_d2, fake_d3)
    t_pred= time.time()-t_inicial
    t.append(t_pred)

    out = fake_b 
    out_img = out[0].detach().squeeze(0).to(device)
    gt_image = gt.cpu().numpy().squeeze()

    ssim = calculate_ssim(gt_image, out_img)
    psnr, mse = calculate_psnr(gt_image, out_img)
    # print("SSIM: ", ssim) 
    # print("PSNR: ", psnr)
    # print("MSE: ", mse)

    # ssims.append(ssim)
    # psnrs.append(psnr)
    # mses.append(mse)

    # if "rain" in str(filename):
    #     rain_mse_list.append(mse)
    #     rain_psnr_list.append(psnr)
    #     rain_ssim_list.append(ssim)


    # if "snow" in str(filename):
    #     snow_mse_list.append(mse)
    #     snow_psnr_list.append(psnr)
    #     snow_ssim_list.append(ssim)

    # if "haze" in str(filename):
    #     haze_mse_list.append(mse)
    #     haze_psnr_list.append(psnr)
    #     haze_ssim_list.append(ssim)

    #Guardar solo 5 imágenes de cada tipo
    if any(saved_img in filename for saved_img in saved_images):
        save_img(out_img, os.path.join(save_img_path, filename[0]))

print('##################### Testing Successfully Completed###################################')    
# with open(file, "w") as results_file:
#     results_file.write(f"Resultados finales:\n")
#     results_file.write(f"\nTotal time: {np.sum(t)}\n")
#     results_file.write(f"Avg Time per image: {np.mean(t)}\n")
#     results_file.write(f"Avg MSE: {np.mean(mses)}\n")
#     results_file.write(f"Avg PSNR: {np.mean(psnrs)}\n")
#     results_file.write(f"Avg SSIM: {np.mean(ssims)}\n")

#     results_file.write("\n Imágenes de lluvia \n")
#     results_file.write(f"MSE medio lluvia: {np.mean(rain_mse_list)}\n")
#     results_file.write(f"PSNR medio lluvia: {np.mean(rain_psnr_list)}\n")
#     results_file.write(f"SSIM medio lluvia: {np.mean(rain_ssim_list)}\n")

#     results_file.write("\n Imágenes de nieve \n")
#     results_file.write(f"MSE medio nieve: {np.mean(snow_mse_list)}\n")
#     results_file.write(f"PSNR medio nieve: {np.mean(snow_psnr_list)}\n")
#     results_file.write(f"SSIM medio nieve: {np.mean(snow_ssim_list)}\n")

#     results_file.write("\n Imágenes de niebla \n")
#     results_file.write(f"MSE medio niebla: {np.mean(haze_mse_list)}\n")
#     results_file.write(f"PSNR medio niebla: {np.mean(haze_psnr_list)}\n")
#     results_file.write(f"SSIM medio niebla: {np.mean(haze_ssim_list)}\n")

print('Restored Results are saved in the results folder')
