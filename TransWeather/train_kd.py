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
from tqdm import tqdm
from collections import OrderedDict

from transweather_model import Transweather
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


#####  Cargar teacher y student #####
teacher = Transweather().to(device)
teacher =  nn.DataParallel(teacher, device_ids=device_ids)
state_dict = torch.load('best_model')
teacher.load_state_dict(state_dict)

student = Student_Transweather()
student = nn.DataParallel(student, device_ids=device_ids)
teacher.eval()
student.train()
####### ------------------------------ #######

date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_folder_path = f"student_checkpoints/{date}"
if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path) 

# --- Load training data and validation/test data --- #

labeled_name = './data/train/allweather.txt'
val_filename1 = './data/test/allweather.txt'


# --- Load training data and validation/test data --- #
train_data_loader = DataLoader(TrainData(crop_size, train_data_dir,labeled_name), batch_size=train_batch_size, shuffle=True, num_workers=8)
val_data_loader = DataLoader(ValData(val_data_dir,val_filename1), batch_size=val_batch_size, shuffle=False, num_workers=8)

mse_loss = torch.nn.MSELoss()

loss_network = LossNetwork(teacher)
loss_network.eval()
optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)

def train_student(student, teacher, optimizer, train_loader, device, T=2):
    total_loss_train = 0.0
    psnr_list = []
    ssim_list = []
    adjust_learning_rate(optimizer, epoch)
    for batch_id, train_data in enumerate(train_loader):
        input_image, gt, imgid = train_data
        input_image = input_image.to(device)
        gt = gt.to(device)
        optimizer.zero_grad() #Antes loss.backward!
        
        with torch.no_grad():
            teacher_pred_image = teacher(input_image)
        student_pred_image = student(input_image)

        soft_targets = nn.functional.softmax(teacher_pred_image / T, dim=-1)
        soft_prob = nn.functional.log_softmax(student_pred_image / T, dim=-1)

        # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
        soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

        # Calculate the true label loss
        label_loss = mse_loss(student_pred_image, gt)
        loss = 0.25 * soft_targets_loss + 0.75 * label_loss
        total_loss_train += loss.item()
       
        loss.backward()
        optimizer.step()

        psnr_list.extend(to_psnr(student_pred_image, gt))
        ssim_list.extend(to_ssim_skimage(student_pred_image, gt))
        

    avr_loss =  total_loss_train/len(train_loader)
    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_loss, avr_psnr, avr_ssim

def eval_student(student, teacher, val_data_loader, device, T=2):
    total_loss_eval = 0.0
    psnr_list = []
    ssim_list = []
    for batch_id, val_data in enumerate(val_data_loader):
        input_image, gt, imgid = val_data
        input_image = input_image.to(device)
        gt = gt.to(device)

        with torch.no_grad():
            teacher_pred_image = teacher(input_image)
            student_pred_image = student(input_image)
            soft_targets = nn.functional.softmax(teacher_pred_image / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_pred_image / T, dim=-1)

            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)
            label_loss = mse_loss(student_pred_image, gt)
            loss = 0.25 * soft_targets_loss + 0.75 * label_loss
            psnr_list.extend(to_psnr(student_pred_image, gt))
            ssim_list.extend(to_ssim_skimage(student_pred_image, gt))

            total_loss_eval += loss
            if not (batch_id % 100):
               print('Epoch: {0}, Iteration: {1}'.format(epoch, batch_id))

    avr_loss_eval = total_loss_eval/len(val_data_loader)
    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_loss_eval, avr_psnr, avr_ssim

old_val_loss = 0
train_losses = []
eval_losses = []
eval_psnrs = []
eval_ssims = []
train_ssims = []
train_psnrs = []
t =[]


for epoch in tqdm(range(epoch_start,num_epochs)):
    t_inicial = time.time()
    student.train()
    train_loss, train_psnr, train_ssim = train_student(student, teacher, optimizer, train_data_loader, device)
    student.eval()
    eval_loss, eval_psnr, eval_ssim = eval_student(student, teacher, val_data_loader, device)
    torch.save(student.state_dict(), '{}/student_epoch{}_loss'.format(save_folder_path, epoch))
    
    train_psnrs.append(train_psnr)
    eval_psnrs.append(eval_psnr)
    eval_ssims.append(eval_ssim)
    train_ssims.append(train_ssim)
    train_losses.append(train_loss)
    eval_losses.append(eval_loss)
    print(f'Epoch {epoch+1}, Training Loss: {train_loss}, Evaluation Loss: {eval_loss}')

    if eval_loss >= old_val_loss:
        torch.save(student.state_dict(), '{}/BEST_loss{}'.format(save_folder_path, eval_loss))
        print('Best model saved')
        old_val_loss = eval_loss

    t_pred= time.time()-t_inicial
    t.append(t_pred)
    print_log(epoch+1, num_epochs, t_pred, train_psnr, train_ssim, eval_psnr, eval_ssim, exp_name, date) 
    print(f"Total time: {np.array(t).sum()}")

with open(f"results_file_{date}", "w") as results_file:
    results_file.write(f"Total time: {sum(t)}\n")
    results_file.write(f"Average time per image: {sum(t)/len(t)}\n")
    results_file.write(f"\n Training results:\n")
    results_file.write(f"Average loss: {sum(train_losses)/len(train_losses)}\n")
    results_file.write(f"Average PSNR: {sum(train_psnrs)/len(train_psnrs)}\n")
    results_file.write(f"Average SSIM: {sum(train_ssims)/len(train_ssims)}\n")
    results_file.write(f"\n Evaluation results:\n")
    results_file.write(f"Average loss: {sum(eval_losses)/len(eval_losses)}\n")
    results_file.write(f"Average PSNR: {sum(eval_psnrs)/len(eval_psnrs)}\n")
    results_file.write(f"Average SSIM: {sum(eval_ssims)/len(eval_ssims)}\n")

    for i in range(len(train_losses)):
        results_file.write(f"\nEpoch {i}\n")
        results_file.write(f"Training Loss: {train_losses[i]}\n")
        results_file.write(f"Eval Loss: {eval_losses[i]}\n")
        results_file.write(f"Training PSNR: {train_psnrs[i]}\n")
        results_file.write(f"Eval PSNR: {eval_psnrs[i]}\n")
        results_file.write(f"Eval SSIM: {eval_ssims[i]}\n")
        results_file.write(f"Training SSIM: {train_ssims[i]}\n")


eval_losses = [loss.cpu().item() for loss in eval_losses]
# train_losses = [loss.cpu().item() for loss in train_losses]
# Graficar los resultados
plt.figure(figsize=(10, 5))
plt.plot(np.array(train_losses), label='Training loss')
plt.plot(np.array(eval_losses), label='Evaluation loss')
plt.title('Training and Evaluation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"student_model_epoch_{date}.png")
plt.show()