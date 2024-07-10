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

from Student_transweather_model import Student_Transweather

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


# --- Define the network --- #
net = Student_Transweather()


# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)


# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
# vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
for param in vgg_model.parameters():
    param.requires_grad = False

# --- Load the network weight --- #
# if os.path.exists('./{}/'.format(exp_name))==False:     
#     os.mkdir('./{}/'.format(exp_name))  
# try:
#     net.load_state_dict(torch.load('./{}/best'.format(exp_name)))
#     print('--- weight loaded ---')
# except:
print('--- no weight loaded ---')


# pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
# print("Total_params: {}".format(pytorch_total_params))
loss_network = LossNetwork(vgg_model)
loss_network.eval()

# --- Load training data and validation/test data --- #

### The following file should be placed inside the directory "./data/train/"
labeled_name = './data/train/allweather.txt'

### The following files should be placed inside the directory "./data/test/"

# val_filename = 'val_list_rain800.txt'
val_filename1 = './data/test/allweather.txt'
# val_filename2 = 'test1.txt'

# --- Load training data and validation/test data --- #
lbl_train_data_loader = DataLoader(TrainData(crop_size, train_data_dir,labeled_name), batch_size=train_batch_size, shuffle=True, num_workers=8)
val_data_loader1 = DataLoader(ValData(val_data_dir,val_filename1), batch_size=val_batch_size, shuffle=False, num_workers=8)
# val_data_loader2 = DataLoader(ValData(val_data_dir,val_filename2), batch_size=val_batch_size, shuffle=False, num_workers=8)


# --- Previous PSNR and SSIM in testing --- #
net.eval()
date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_folder_path = './student_checkpoints/LightModel/{}'.format(date)
if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
net.train()

def eval(net, val_data_loader, device, save_tag=False):
    """
    :param net: GateDehazeNet
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param category: indoor or outdoor test dataset
    :param save_tag: tag of saving image or not
    :return: average PSNR value
    """
    total_loss_eval = 0.0
    psnr_list = []
    ssim_list = []
    total_batches_eval = len(val_data_loader)
    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            haze, gt, image_name = val_data          
            # print(haze.shape, gt.shape)  
            haze = haze.to(device)
            gt = gt.to(device)
            dehaze, _, _ = net(haze)

        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(dehaze, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(to_ssim_skimage(dehaze, gt))
        total_loss_eval += loss
        if save_tag:
            save_image(dehaze, image_name)

    avr_loss_eval = total_loss_eval/len(val_data_loader)
    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_loss_eval, avr_psnr, avr_ssim


old_val_loss = 0
train_losses = []
eval_losses = []
eval_psnrs = []
eval_ssims = []
train_psnrs = []
train_ssims = []
t =[]

# Early stopping variables
best_eval_loss = float('inf')
epochs_without_improvement = 0
early_stopping_patience = 10

for epoch in tqdm(range(epoch_start,num_epochs)):
    t_inicial = time.time()
    total_loss_train = 0.0
    psnr_list = []
    ssim_list = []
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch)
#-------------------------------------------------------------------------------------------------------------
    for batch_id, train_data in enumerate(lbl_train_data_loader):
        input_image, gt, imgid = train_data
        input_image = input_image.to(device)
        gt = gt.to(device)
        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()
        # --- Forward + Backward + Optimize --- #
        net.train()
        pred_image, _, _ = net(input_image)

        smooth_loss = F.smooth_l1_loss(pred_image, gt)
        perceptual_loss = loss_network(pred_image, gt)

        loss = smooth_loss + lambda_loss*perceptual_loss 
        total_loss_train += loss.item()

        loss.backward()
        optimizer.step()
        
        # --- To calculate average PSNR --- #
        psnr_list.extend(to_psnr(pred_image, gt))
        ssim_list.extend(to_ssim_skimage(pred_image, gt))

        if not (batch_id % 100):
            print('Epoch: {0}, Iteration: {1}'.format(epoch, batch_id))

    # --- Calculate the average training PSNR in one epoch --- #
    avr_train_loss =  total_loss_train/len(lbl_train_data_loader)
    avr_train_psnr = sum(psnr_list) / len(psnr_list)
    avr_train_ssim = sum(ssim_list) / len(ssim_list)

    # --- Use the evaluation model in testing --- #
    net.eval()
    # val_psnr, val_ssim = validation(net, val_data_loader, device, exp_name)
    eval_loss, eval_psnr, eval_ssim = eval(net, val_data_loader1, device, exp_name)
    # val_psnr2, val_ssim2 = validation(net, val_data_loader2, device, exp_name)
    torch.save(net.state_dict(), '{}/student_epoch{}_loss{}'.format(save_folder_path, epoch, eval_loss))

    train_psnrs.append(avr_train_psnr)
    eval_psnrs.append(eval_psnr)
    eval_ssims.append(eval_ssim)
    train_ssims.append(avr_train_ssim)
    train_losses.append(avr_train_loss)
    eval_losses.append(eval_loss)
    one_epoch_time = time.time() - start_time

    # --- update the network weight --- #
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        epochs_without_improvement = 0
        torch.save(net.state_dict(), '{}/BEST_loss{}'.format(save_folder_path, eval_loss))
        print('Best model saved')
    else:
        epochs_without_improvement += 1

    t_pred= time.time()-t_inicial
    t.append(t_pred)
    print_log(epoch+1, num_epochs, t_pred, avr_train_psnr, avr_train_ssim, eval_psnr, eval_ssim, exp_name, date) 
    print(f"Total time: {np.array(t).sum()}")

    if epochs_without_improvement >= early_stopping_patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break
    

eval_losses = [loss.cpu().item() for loss in eval_losses]
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training loss')
plt.plot(eval_losses, label='Evaluation loss')
plt.title('Training and Evaluation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"LightModel_{date}.png")
plt.show()