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
#from transweather_model import Transweather
from Student_transweather_model import Student_Transweather

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-model', help='path to model', default='2024-06-05_12-41-14/student_epoch1_loss1.4482010965853667')
parser.add_argument('-save', help='save path', default='pruebasKD')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
args = parser.parse_args()

val_batch_size = args.val_batch_size
exp_name = args.exp_name
save_path =('./' + str(args.save))

#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

# --- Set category-specific hyper-parameters  --- #
val_data_dir = ''

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# --- Validation data loader --- #

val_filename = './data/test/allweather.txt' ## This text file should contain all the names of the images and must be placed in ./data/test/ directory

val_data_loader = DataLoader(ValData(val_data_dir,val_filename), batch_size=val_batch_size, shuffle=False, num_workers=8)

# --- Define the network --- #

net = Student_Transweather().cuda()
# net = Student_Transweather().cuda()

net = nn.DataParallel(net, device_ids=device_ids)


# --- Load the network weight --- #
model = './student_checkpoints/' + str(args.model)
net.load_state_dict(torch.load(model))
print(f"Utilizando modelo {model[22:]}")
print(f"Guardando imágenes en {save_path}")
# --- Use the evaluation model in testing --- #
net.eval()
category = "test1"

if os.path.exists(save_path)==False:     
    os.makedirs(save_path)   


def save_test_image(dehaze, image_name):
    dehaze_images = torch.split(dehaze, 1, dim=0)
    batch_num = len(dehaze_images)    
    for ind in range(batch_num):
        utils.save_image(dehaze_images[ind], '{}/{}'.format(save_path, os.path.basename(image_name[ind])[:-3] + 'png'))

def test(net, val_data_loader, device, save_tag=False):
    """
    :param net: GateDehazeNet
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param category: indoor or outdoor test dataset
    :param save_tag: tag of saving image or not
    :return: average PSNR value
    """
    psnr_list = []
    ssim_list = []
    total_batches_eval = len(val_data_loader)
    total_loss_val = 0
    for batch_id, val_data in tqdm(enumerate(val_data_loader)):

        with torch.no_grad():
            haze, gt, image_name = val_data          
            # print(haze.shape, gt.shape)  
            haze = haze.to(device)
            gt = gt.to(device)
            # print(haze.size())
            dehaze = net(haze)
            # pdb.set_trace()
            
        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(dehaze, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(to_ssim_skimage(dehaze, gt))
        # import ipdb
        # ipdb.set_trace()
        # --- Save image --- #
        if save_tag:
            save_test_image(dehaze, image_name)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim

print('--- Testing starts! ---')
start_time = time.time()
val_psnr, val_ssim = test(net, val_data_loader, device, save_tag=True)

end_time = time.time() - start_time
print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
print('validation time is {0:.4f}'.format(end_time))
