
import argparse
import os
from dataset_test import DatasetFromFolder_Test
import torch
from torch.utils.data import DataLoader
from os.path import join
from utils import is_image_file, load_img, save_img
torch.backends.cudnn.benchmark = True
from metrics import *
import time
import ast 
import torch
import onnxruntime as ort
import onnx
import time
import numpy as np


'''
AVISO: 
Puede que de problemas de que falta un input, hay que quita o poner la _ en input_1 de out
'''

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', default='../smaller_test', required=False, help='facades')
parser.add_argument('--save_path', default='results/KD_onnx', required=False, help='facades')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--cuda', default= True, action='store_false',  help='use cuda')
opt = parser.parse_args()
print(opt)

tiempos_imagenes = {}

type_net = "og" ## Elegir entre: og (original), fp16, int8

onnx_path = './onnx_models'

onnx.load(f'{onnx_path}/domain_gpu.onnx')
ort_sess = ort.InferenceSession(f'{onnx_path}/domain_gpu.onnx')

d1_path = f'{onnx_path}/{type_net}/domain1_{type_net}.onnx'
d2_path = f'{onnx_path}/{type_net}/domain2_{type_net}.onnx'
d3_path = f'{onnx_path}/{type_net}/domain3_{type_net}.onnx'

onnx.load(d1_path)
ort_sess1 = ort.InferenceSession(d1_path)
onnx.load(d2_path)
ort_sess2 = ort.InferenceSession(d2_path)
onnx.load(d3_path)
ort_sess3 = ort.InferenceSession(d3_path)

print(f"Usando modelo {type_net}")

opt.save_path = opt.save_path+f'{type_net}'
if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Guardando en', opt.save_path)


def get_test_set(root_dir):
    test_dir = join(root_dir)

    return DatasetFromFolder_Test(test_dir)

test_set = get_test_set(opt.dataset+"/input")
testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.test_batch_size, shuffle=False)


gt_set=get_test_set(opt.dataset+"/gt")
gt_data_loader = DataLoader(dataset=gt_set, batch_size=opt.test_batch_size, shuffle=False)


all_times = []
save_img_path = f"results/Cuantificación/{type_net}" #Carpeta donde se guardan las imágenes
if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)

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

print(len(testing_data_loader))
for iteration_test, batch in enumerate(zip(testing_data_loader, gt_data_loader)):
    real_a, filename = batch[0][0].to(device), batch[0][1]
    gt, gt_filename = batch[1][0].to(device), batch[1][1]
    print(filename)
    real_a = real_a.to(device)
    gt_path = "gt/"+str(filename)

    if type_net == "fp16":
        t_inicial = time.time() 
        fake_d1 = ort_sess1.run(None, {'modelInput': np.float16(real_a.cpu().numpy())})
        fake_d2 = ort_sess2.run(None, {'modelInput': np.float16(real_a.cpu().numpy())})
        fake_d3 = ort_sess3.run(None, {'modelInput': np.float16(real_a.cpu().numpy())})

        out = ort_sess.run(None,{'input1': np.float16(real_a.cpu().numpy()), 'input_2': np.float16(fake_d1[0]), 'input_3': np.float16(fake_d2[0]), 'input_4': np.float16(fake_d3[0])})
        t_pred = time.time() - t_inicial
        t.append(t_pred)
    else:
        t_inicial = time.time() 
        fake_d1 = ort_sess1.run(None, {'modelInput': real_a.cpu().numpy()})
        fake_d2 = ort_sess2.run(None, {'modelInput': real_a.cpu().numpy()})
        fake_d3 = ort_sess3.run(None, {'modelInput': real_a.cpu().numpy()})
 
        out = ort_sess.run(None,{'input_1': real_a.cpu().numpy(), 'input_2':fake_d1[0],'input_3':fake_d2[0], 'input_4':fake_d3[0]})
        t_pred = time.time()-t_inicial
        t.append(t_pred)

    out_img = out[0].squeeze(0)
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

    save_img(out_img, os.path.join(save_img_path, filename[0]))
    

with open(f'resultados_KD_{type_net}', "w") as results_file:
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