import argparse
import os
from os.path import exists, join as join_paths
import torch
import numpy as np
import warnings
from tqdm import tqdm

from torchvision.utils import save_image,make_grid
from torch.utils.data import DataLoader

from dataloader_udr import *
from metrics import *
from psnr_ssim import *
from Teacher_UDR import *

warnings.filterwarnings("ignore")

#Modificar model path en parser

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--tile', type=int, default=320, help='Tile size, None for no tile during testing (testing as a whole)')
parser.add_argument('--tile_overlap', type=int, default=64, help='Overlapping of different tiles')
parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8') 
parser.add_argument('--dataset_type', type=str, default='raindrop_syn') 
parser.add_argument('--dataset_raindrop_syn', type=str, default='../data/smaller_test/input/', help='path of syn dataset')
parser.add_argument('--savepath', type=str, default='out/Pruning', help='path of output image') 
parser.add_argument('--model_path', type=str, default='udrs2former_raindrop_real.pth', help='path of checkpoint') 


opt = parser.parse_args()

if opt.dataset_type == 'raindrop_syn':
    rain_test = DataLoader(dataset=RainDS_Dataset(opt.dataset_raindrop_syn, train=True), batch_size=1, shuffle=False, num_workers=8)
netG_1 = Transformer(img_size=(opt.tile,opt.tile)).cuda()

saved_images = ["17_R_rain.jpg", '25_R_rain.jpg', '24_rain.png', '130_rain.png', '123_rain.png', 'haze_0003_s100_a04.png', 'haze_0005_s100_a04.png', 'haze_0011_s80_a05.png', 'haze_0013_s85_a06.png', 'haze_0014_s80_a04.png', 'snow_00018.jpg', 'snow_00075.jpg', 'snow_00148.jpg', 'snow_00202.jpg', 'snow_00270.jpg']


if __name__ == '__main__':   
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

    model_path = opt.model_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    ckpt = torch.load(model_path, map_location=device)
    netG_1.load_state_dict(ckpt)

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

    net_pruned = prune_model(netG_1, amount=0.3)

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

    import torch.nn.functional as F

    def resize_and_pad(tensor, size):
        _, _, h, w = tensor.size()
        target_h, target_w = size

        # Calculate aspect ratio
        aspect_ratio = w / h

        if aspect_ratio > 1:
            # Width is greater than height, resize based on width
            new_w = target_w
            new_h = int(target_w / aspect_ratio)
        else:
            # Height is greater than width, resize based on height
            new_h = target_h
            new_w = int(target_h * aspect_ratio)

        # Resize the tensor
        tensor_resized = F.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # Pad the tensor to the target size
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2

        tensor_padded = F.pad(tensor_resized, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)

        return tensor_padded

    savepath_dataset = opt.savepath

    if not os.path.exists(savepath_dataset):
        os.makedirs(savepath_dataset)
        
    loop = tqdm(enumerate(rain_test),total=len(rain_test))
    with torch.no_grad():
        for idx,(haze,clean,name) in loop:
            try:
                #print(haze.shape)
                #if haze.shape == [1,3,268,400]:
                #    haze_resized_padded = resize_and_pad(haze, (480,720))
                haze = haze.cuda();clean = clean.cuda()
                #print(haze.shape)
                b, c, h, w = haze.size()
                tile = min(opt.tile, h, w)
                tile_overlap = opt.tile_overlap
                sf = opt.scale

                stride = tile - tile_overlap
                h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
                w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
                E1 = torch.zeros(b, c, h*sf, w*sf).type_as(haze)
                W1 = torch.zeros_like(E1)
                E2 = torch.zeros(b, c, h*sf, w*sf).type_as(haze)
                W2 = torch.zeros_like(E2)
                
                t_inicial = time.time() 
                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        print(f"Processing patch at w_idx: {w_idx}, h_idx: {h_idx}, image name: {name}")
                        in_patch = haze[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                        '''
                        expected_size = (net_pruned.img_size[0], net_pruned.img_size[1])  # Ensure your model has this attribute
                        print(expected_size)
                        if in_patch.shape[-2:] != expected_size:
                            in_patch = F.interpolate(in_patch, size=expected_size, mode='bilinear', align_corners=False)
                        '''
                        out_patch1,_ = net_pruned(in_patch)
                        out_patch1 = out_patch1[0]
                        out_patch_mask1 = torch.ones_like(out_patch1)
                        E1[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch1)
                        W1[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask1)
                dehaze = E1.div_(W1)
                t_pred = time.time()-t_inicial
            except:
                continue

            t.append(t_pred)
            
            save_image(dehaze, os.path.join(savepath_dataset,'%s.png'%(name)),normalize=False)

            ssim1=calculate_ssim(dehaze*255,clean*255,crop_border=0,test_y_channel=True)
            psnr1,mse1=calculate_psnr(dehaze*255,clean*255,crop_border=0,test_y_channel=True)

            ssims.append(ssim1)
            psnrs.append(psnr1)
            mses.append(mse1)

            print("Image saved in: ", os.path.join(savepath_dataset,'%s.png'%(name)))
            print('Generated images %04d of %04d' % (idx+1, len(rain_test)))


    ssim = np.mean(ssims)
    psnr = np.mean(psnrs)
    mse = np.mean(mses)
    print('ssim_avg:',ssim)
    print('psnr_avg:',psnr)
    print('mse_avg:',mse)

    #Guarda resultados en listas globales
    ssims.append(ssim)
    psnrs.append(psnr)
    mses.append(mse)

    #Guarda los resutlados en las listas específicas de climas
    if "rain" in str(name):
        rain_mse_list.append(mse)
        rain_psnr_list.append(psnr)
        rain_ssim_list.append(ssim)

    if "snow" in str(name):
        snow_mse_list.append(mse)
        snow_psnr_list.append(psnr)
        snow_ssim_list.append(ssim)

    if "haze" in str(name):
        haze_mse_list.append(mse)
        haze_psnr_list.append(psnr)
        haze_ssim_list.append(ssim)


    #Guarda resultados en el fichero
    with open(os.path.join(opt.savepath,opt.dataset_type, model_path, "results_file"), "w") as results_file:
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
 
