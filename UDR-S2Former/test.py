import argparse
import os
from os.path import exists, join as join_paths
import torch
import numpy as np
import warnings
from tqdm import tqdm

from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader

from dataloader_udr import *
from metrics import *
from psnr_ssim import *
from Student_UDR import *
import time

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--tile', type=int, default=320, help='Tile size, None for no tile during testing (testing as a whole)')
parser.add_argument('--tile_overlap', type=int, default=64, help='Overlapping of different tiles')
parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8') 
parser.add_argument('--dataset_type', type=str, default='raindrop_syn') 
parser.add_argument('--dataset_raindrop_syn', type=str, default='RainDS/RainDS_syn/', help='path of syn dataset')
parser.add_argument('--savepath', type=str, default='./out/', help='path of output image') 
parser.add_argument('--model_path', type=str, default='pretrained/udrs2former_', help='path of checkpoint') 


opt = parser.parse_args()

if opt.dataset_type == 'raindrop_syn':
    rain_test = DataLoader(dataset=RainDS_Dataset(opt.dataset_raindrop_syn, train=True), batch_size=1, shuffle=False, num_workers=4)
netG_1 = Transformer_Student(img_size=(opt.tile,opt.tile)).cuda()

if __name__ == '__main__':   
    ssims = []
    psnrs = []
    rmses = []
    mses = []
    t=[]

    # opt.model_path = opt.model_path + opt.dataset_type + '.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # g1ckpt1 = opt.model_path
    #ckpt = torch.load(g1ckpt1, map_location=device)
    model_path = 'pretrained/2024-05-08_14-24-42/student_model_epoch_200_2024-05-09_05-22-45.pth'
    ckpt = torch.load(model_path, map_location=device)
    netG_1.load_state_dict(ckpt)

    savepath_dataset = os.path.join(opt.savepath,opt.dataset_type, model_path)

    if not os.path.exists(savepath_dataset):
        os.makedirs(savepath_dataset)
        
    loop = tqdm(enumerate(rain_test),total=len(rain_test))
    for idx,(haze,clean,name) in loop:
        with torch.no_grad():
                
                haze = haze.cuda();clean = clean.cuda()
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
                        in_patch = haze[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                        out_patch1,_ = netG_1(in_patch)
                        out_patch1 = out_patch1[0]
                        out_patch_mask1 = torch.ones_like(out_patch1)
                        E1[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch1)
                        W1[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask1)
                dehaze = E1.div_(W1)
                t_pred = time.time()-t_inicial
                t.append(t_pred)

                save_image(dehaze, os.path.join(savepath_dataset,'%s.png'%(name)),normalize=False)


                ssim1=calculate_ssim(dehaze*255,clean*255,crop_border=0,test_y_channel=True)
                psnr1,mse1=calculate_psnr(dehaze*255,clean*255,crop_border=0,test_y_channel=True)

                ssims.append(ssim1)
                psnrs.append(psnr1)
                mses.append(mse1)

                print("Image saved in: ", os.path.join(savepath_dataset,'%s.png'%(name)))
                print('Generated images %04d of %04d' % (idx+1, len(rain_test)))
                print('ssim:',(ssim1))
                print('psnr:',(psnr1))
                print('mse:',(mse1))

        ssim = np.mean(ssims)
        psnr = np.mean(psnrs)
        mse = np.mean(mses)
        print('ssim_avg:',ssim)
        print('psnr_avg:',psnr)
        print('mse_avg:',mse)

    with open(os.path.join(opt.savepath,opt.dataset_type, model_path, "results_file"), "w") as results_file:
        results_file.write(f"\nResultados finales:\n")
        results_file.write(f"Total time: {np.sum(t)}\n")

        results_file.write(f"\nResultados generales\n")
        results_file.write(f"Avg SSIM: {ssim}\n")
        results_file.write(f"Avg PSNR: {psnr}\n")
        results_file.write(f"Avg MSE: {mse}\n")
        results_file.write(f"Avg Time: {np.mean(t)}\n")
 
