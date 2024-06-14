import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_data_functions import TrainData
from val_data_functions import ValData
from utils import *
from torchvision.models import vgg16
from perceptual import LossNetwork
import os
import numpy as np
import random
from datetime import datetime

from transweather_model import Transweather

plt.switch_backend('agg')

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=2e-4, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[256, 256], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=64, type=int)
parser.add_argument('-epoch_start', help='Starting epoch number of the training', default=0, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=64, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', default="exp1", type=str)
parser.add_argument('-seed', help='set random seed', default=42, type=int)
parser.add_argument('-num_epochs', help='number of epochs', default=200, type=int)

args = parser.parse_args()

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
epoch_start = args.epoch_start
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
exp_name = args.exp_name
num_epochs = args.num_epochs


#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nlambda_loss: {}'.format(learning_rate, crop_size,
      train_batch_size, val_batch_size, lambda_loss))


train_data_dir = ''
val_data_dir = ''

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Transweather()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# --- Multi-GPU --- #
net = net.to(device)
# net = nn.DataParallel(net, device_ids=device_ids)


# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
# vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
for param in vgg_model.parameters():
    param.requires_grad = False

# --- Load the network weight --- #
if os.path.exists('./{}/'.format(exp_name))==False:     
    os.mkdir('./{}/'.format(exp_name))  
# try:
#     net.load_state_dict(torch.load('./{}/best'.format(exp_name)))
#     print('--- weight loaded ---')
# except:
#     print('--- no weight loaded ---')

loss_network = LossNetwork(vgg_model)
loss_network.eval()
labeled_name = './data/train/allweather.txt'

val_filename1 = './data/test/allweather.txt'
lbl_train_data_loader = DataLoader(TrainData(crop_size, train_data_dir,labeled_name), batch_size=train_batch_size, shuffle=True, num_workers=8)
val_data_loader1 = DataLoader(ValData(val_data_dir,val_filename1), batch_size=val_batch_size, shuffle=False, num_workers=8)

date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

save_folder_path = './pretrained/{}'.format(date)
if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
net.train()
old_val_psnr1 = 0
val_psnr_list = []


def training(net, optimizer, lbl_train_data_loader, device):
    psnr_list = []
    adjust_learning_rate(optimizer, epoch)
    total_loss_train = 0
    total_batches_train = len(lbl_train_data_loader)
    
    for batch_id, train_data in enumerate(lbl_train_data_loader):

        input_image, gt, imgid = train_data
        input_image = input_image.to(device)
        gt = gt.to(device)
        optimizer.zero_grad()
        net.train()
        pred_image = net(input_image)

        smooth_loss = F.smooth_l1_loss(pred_image, gt)
        perceptual_loss = loss_network(pred_image, gt)

        loss = smooth_loss + lambda_loss*perceptual_loss 

        loss.backward()
        optimizer.step()

        # --- To calculate average PSNR --- #
        total_loss_train += loss.item()
        psnr_list.extend(to_psnr(pred_image, gt))

        if not (batch_id % 100):
            print('Epoch: {0}, Iteration: {1}'.format(epoch, batch_id))

    # --- Calculate the average training PSNR in one epoch --- #
    avr_train_psnr = sum(psnr_list) / len(psnr_list)
    avr_loss_train = total_loss_train / total_batches_train
    
    return avr_loss_train, avr_train_psnr

def validation(net, val_data_loader, device, save_tag=False):
    psnr_list = []
    ssim_list = []
    total_batches_val = len(val_data_loader)
    total_loss_val = 0

    for batch_id, val_data in tqdm(enumerate(val_data_loader)):

        with torch.no_grad():
            haze, gt, image_name = val_data          
            haze = haze.to(device)
            gt = gt.to(device)
            dehaze = net(haze)
            smooth_loss = F.smooth_l1_loss(dehaze, gt)
            perceptual_loss = loss_network(dehaze, gt)

            loss = smooth_loss + lambda_loss*perceptual_loss 
            
        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(dehaze, gt))
        total_loss_val += loss.item()

        # --- Calculate the average SSIM --- #
        ssim_list.extend(to_ssim_skimage(dehaze, gt))

        # --- Save image --- #
        if save_tag:
            save_image(dehaze, image_name)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    avr_loss_train = total_loss_val / total_batches_val

    return avr_loss_train, avr_psnr, avr_ssim


t = []
epoch_val_loss = []
epoch_train_loss = []
train_losses = []
train_psnrs =[]
val_losses = []
val_psnrs = []
val_ssims = []
t_inicial = time.time() 

for epoch in range(epoch_start,num_epochs):
    start_time = time.time()
    train_loss, train_psnr = training(net, optimizer, lbl_train_data_loader, device)
    train_losses.append(train_loss)
    train_psnrs.append(train_psnr)

    # --- Save the network parameters --- #
    torch.save(net.state_dict(), '{}/epoch{}_loss{}.pth'.format(save_folder_path, epoch, train_loss))
    # --- Use the evaluation model in testing --- #
    net.eval()

    val_psnr1, val_ssim1, val_loss = validation(net, val_data_loader1, device, exp_name)
    val_psnrs.append(val_psnr1)
    val_ssims.append(val_ssim1)
    val_losses.append(val_loss)

    one_epoch_time = time.time() - start_time
    print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr1, val_ssim1, exp_name)

    # --- update the network weight --- #
    if val_psnr1 >= old_val_psnr1:
        torch.save(net.state_dict(), '{}/BEST.pth'.format(save_folder_path))
        print('model saved')
        old_val_psnr1 = val_psnr1

        # Note that we find the best model based on validating with raindrop data. 
    t_pred= time.time()-start_time
    t.append(t_pred) 
    print(f"Total time: {np.sum(t)}")
    
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Evaluation loss')
plt.title('Training and Evaluation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"student_model_epoch_{date}.png")
# plt.show()


