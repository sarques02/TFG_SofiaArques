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
from torch.optim.lr_scheduler import StepLR


from Decoder_transweather_model import Transweather
from Decoder_Student_transweather_model import Student_Transweather

plt.switch_backend('agg')

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('--learning_rate', help='Set the learning rate', default=1e-4, type=float)  # Reduced learning rate
parser.add_argument('--crop_size', help='Set the crop_size', default=[256, 256], nargs='+', type=int)
parser.add_argument('--train_batch_size', help='Set the training batch size', default=64, type=int)
parser.add_argument('--epoch_start', help='Starting epoch number of the training', default=0, type=int)
parser.add_argument('--lambda_loss', help='Set the lambda in loss function', default=0.01, type=float)  # Adjust lambda_loss
parser.add_argument('--val_batch_size', help='Set the validation/test batch size', default=64, type=int)
parser.add_argument('--exp_name', help='directory for saving the networks of the experiment', default="exp1", type=str)
parser.add_argument('--seed', help='set random seed', default=42, type=int)
parser.add_argument('--num_epochs', help='number of epochs', default=100, type=int)  # Reduced epochs for faster experiments

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
state_dict = torch.load('best_model')


#####  Cargar teacher y student #####
teacher = Transweather().to(device)
teacher =  nn.DataParallel(teacher, device_ids=device_ids)
state_dict_teacher = torch.load('best_model')
teacher.load_state_dict(state_dict)

student = Student_Transweather()
state_dict_to_load = {k: v for k, v in state_dict_teacher.items() if not k.startswith('module.Tdec')}
# for k in state_dict_to_load:
#     print(f'Se seleccionó la clave: {k}')
student.load_state_dict(state_dict_to_load, strict=False)
student = nn.DataParallel(student, device_ids=device_ids)
teacher.eval()
student.train()

layer_view = 0
for name, param in student.named_parameters():
    if "Tdec" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
    print(name, param.requires_grad)

date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_folder_path = f"student_checkpoints/Decoder/{date}"
if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path) 

# --- Load training data and validation/test data --- #

labeled_name = './data/train/allweather.txt'
val_filename1 = './data/test/allweather.txt'


# --- Load training data and validation/test data --- #
train_data_loader = DataLoader(TrainData(crop_size, train_data_dir,labeled_name), batch_size=train_batch_size, shuffle=True, num_workers=8)
val_data_loader = DataLoader(ValData(val_data_dir,val_filename1), batch_size=val_batch_size, shuffle=False, num_workers=8)

mse_loss = torch.nn.MSELoss()  # Changed to L1 Loss

loss_network = LossNetwork(teacher)
loss_network.eval()
optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)



def print_log(epoch, num_epochs, one_epoch_time, date, loss, avg_val_loss):
    one_epoch_time = float(one_epoch_time)
    # --- Write the training log --- #
    with open(f"{save_folder_path}/KD_StudentDecoder_log_{date}.txt", 'a') as f:
        print('Epoch: [{2}/{3}], Date: {0}s, Time_Cost: {1:.0f}s, Loss Train: {4}, loss eval: {5}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, loss, avg_val_loss), file=f)

def tenc_loss(student_outputs, teacher_outputs):
    # Verificar que las listas tienen la misma longitud
    assert len(student_outputs) == len(teacher_outputs)

    losses = []
    for student_output, teacher_output in zip(student_outputs, teacher_outputs):
        # Calcular la pérdida entre las salidas de cada capa
        loss = mse_loss(student_output, teacher_output) 
        losses.append(loss)

    # Calcular la pérdida total promediando todas las pérdidas individuales
    total_loss = sum(losses) / len(losses)
    return total_loss

def batch_PSNR(img1, img2, data_range):
    """Calcula el PSNR para un batch de imágenes.
    
    Args:
        img1 (torch.Tensor): Batch de imágenes generadas por el modelo.
        img2 (torch.Tensor): Batch de imágenes de verdad terreno (ground truth).
        data_range (float): El rango de los valores de los datos (por ejemplo, 1.0 si los valores están normalizados entre 0 y 1).
    
    Returns:
        float: PSNR promedio para el batch de imágenes.
    """
    mse = F.mse_loss(img1, img2, reduction='none')
    mse_per_image = mse.view(mse.size(0), -1).mean(dim=1)
    psnr = 10 * torch.log10(data_range**2 / mse_per_image)
    return psnr.mean().item()

def train_student(student, teacher, optimizer, train_loader, device, T=3):
    student.train()
    total_loss_train = 0.0

    for batch_id, train_data in enumerate(train_loader):
        input_image, gt, imgid = train_data
        input_image = input_image.to(device)
        gt = gt.to(device)

        student_pred_image, student_encoder = student(input_image)
        with torch.no_grad():
            teacher_pred_image, teacher_encoder = teacher(input_image)

        soft_targets_loss = mse_loss(student_pred_image, teacher_pred_image)
        tenc_l = tenc_loss(student_encoder, teacher_encoder)
        loss = soft_targets_loss + lambda_loss * tenc_l

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss_train += loss.item()

    return total_loss_train / len(train_loader)


def validate(student, teacher, val_data_loader, device):
    student.eval()
    with torch.no_grad():
        total_psnr = 0.0
        total_val_loss = 0.0
        for batch_id, val_data in enumerate(val_data_loader):
            input_image, gt, imgid = val_data
            input_image = input_image.to(device)
            gt = gt.to(device)

            student_pred_image, _ = student(input_image)
            teacher_pred_image, _ = teacher(input_image)

            val_loss = mse_loss(student_pred_image, gt)
            total_val_loss += val_loss.item()

            psnr = batch_PSNR(student_pred_image, gt, 1.0)  # Usando la función definida
            total_psnr += psnr

        avg_psnr = total_psnr / len(val_data_loader)
        avg_val_loss = total_val_loss / len(val_data_loader)
    return avg_val_loss, avg_psnr

# --- Main training and validation --- #
since = time.time()
train_loss = []
val_loss = []
psnr_list = []

for epoch in range(num_epochs):
    epoch_start = time.time()
    loss = train_student(student, teacher, optimizer, train_data_loader, device)
    avg_val_loss, avg_psnr = validate(student, teacher, val_data_loader, device)
    train_loss.append(loss)
    val_loss.append(avg_val_loss)
    psnr_list.append(avg_psnr)
    one_epoch_time = time.time() - epoch_start

    print('Epoch: [{}/{}], Time_Cost: {:.0f}s, Train Loss: {:.4f}, Validation Loss: {:.4f}, Validation PSNR: {:.4f}'
          .format(epoch + 1, num_epochs, one_epoch_time, loss, avg_val_loss, avg_psnr))

    print_log(epoch + 1, num_epochs, one_epoch_time, date, loss, avg_val_loss)
    torch.save(student.state_dict(), f"{save_folder_path}/Decoderstudent_epoch_{epoch+1}.pth")
torch.save(student.state_dict(), f"{save_folder_path}/Decoderstudent_final.pth")

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


plt.figure(figsize=(10, 5))
plt.plot(np.array(train_loss), label='Training loss')
plt.plot(np.array(val_loss), label='Evaluation loss')
plt.title('Training and Evaluation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"Decoder_{date}.png")
plt.show()
