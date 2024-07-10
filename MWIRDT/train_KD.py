import argparse
import os
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from os import listdir
from os.path import join
import torch.utils.data as data
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as T
import tqdm
import time
import matplotlib.pyplot as plt
from datetime import datetime

import student_network_rain2 as Student

class MultiDatasetFromFolder(Dataset):
    def __init__(self, image_dirs, input_to_gt_ratios, limit=8000):
        super(MultiDatasetFromFolder, self).__init__()
        self.input_filenames = []
        self.gt_filenames = []
        self.input_to_gt_ratios = input_to_gt_ratios
        self.limit = limit
        
        for idx, image_dir in enumerate(image_dirs):
            input_dir = os.path.join(image_dir, 'input')
            gt_dir = os.path.join(image_dir, 'gt')
            ratio = self.input_to_gt_ratios[idx]
            
            gt_images = [f for f in os.listdir(gt_dir) if self.is_image_file(f)]
            gt_images = gt_images[:self.limit]  # Limita a las primeras 8000 imágenes gt
            for gt_image in gt_images:
                gt_path = os.path.join(gt_dir, gt_image)
                self.gt_filenames.append(gt_path)
                
                base_name = os.path.splitext(gt_image)[0]
                input_files = [input_file for input_file in os.listdir(input_dir) if input_file.startswith(base_name) and self.is_image_file(input_file)]
                
                if ratio == 1:
                    input_files = input_files[:1]  # Limita a la primera imagen input si ratio es 1:1
                else:
                    input_files = input_files[:min(len(input_files), ratio)]  # Limita a la relación especificada
                
                for input_file in input_files:
                    if len(self.input_filenames) < self.limit * ratio:
                        self.input_filenames.append(os.path.join(input_dir, input_file))

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp", ".tif"])

    def __getitem__(self, index):
        ratio_index = 0
        current_index = index

        for ratio in self.input_to_gt_ratios:
            dataset_size = len(self.gt_filenames) * ratio
            if current_index < dataset_size:
                ratio_index = ratio
                break
            else:
                current_index -= dataset_size
        
        gt_index = current_index // ratio_index
        input_index = current_index

        gt_image_path = self.gt_filenames[gt_index]
        input_image_path = self.input_filenames[input_index]

        input_image = cv2.imread(input_image_path)
        gt_image = cv2.imread(gt_image_path)

        input_image = cv2.resize(input_image, (256, 256), interpolation=cv2.INTER_CUBIC)
        gt_image = cv2.resize(gt_image, (256, 256), interpolation=cv2.INTER_CUBIC)

        input_image = transforms.ToTensor()(input_image)    
        input_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(input_image)

        gt_image = transforms.ToTensor()(gt_image)    
        gt_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(gt_image)

        return input_image, gt_image

    def __len__(self):
        return min(len(self.input_filenames), self.limit)

batch_size = 32

### -------- Data Loaders ----------- ###

print('Loading data...')

train_image_dirs = [
    "./dataset/train/CSD/",
    "./dataset/train/Rain13K/",
    "./dataset/train/OTS/"
]

test_image_dirs = [
    "./dataset/test/CSD/",
    "./dataset/test/Rain13K/",
    "./dataset/train/OTS/"
]

input_to_gt_ratios_train = [1, 1, 35]  # Define la relación input-to-gt para cada dataset de entrenamiento
input_to_gt_ratios_test = [1, 1, 35]   # Define la relación input-to-gt para cada dataset de prueba


train_set = MultiDatasetFromFolder(train_image_dirs, input_to_gt_ratios_train)
training_data_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

test_set = MultiDatasetFromFolder(test_image_dirs, input_to_gt_ratios_test)
testing_data_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

# ----------------------------------------#
print("Data loaded")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net_num = 2 #1,2,3
factor_num= 8 #4,8
d3_path = f"./checkpoints/domain{net_num}.pth" #1 niebla, 2 lluvia, 3 nieve
save_folder = f'./models/net{net_num}/factor{factor_num}'
teacher = torch.load(d3_path)
teacher.eval().to(device)

student = Student.My_net()
student.train().to(device)


def print_log(epoch, num_epochs, one_epoch_time, date, net_num = net_num, factor_num = factor_num):
    one_epoch_time = float(one_epoch_time)

       # --- Write the training log --- #
    with open(f"{save_folder}/log_{date}_net{net_num}_factor{factor_num}.txt", 'a') as f:
        print('Epoch: [{2}/{3}], Date: {0}s, Time_Cost: {1:.0f}s'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs), file=f)


mse_loss = torch.nn.MSELoss()

def train_student(student, teacher, optimizer, train_loader, device):
    student.train()
    total_loss = 0.0
    total_batches = len(train_loader)

    for input_img, gt_img in train_loader:
        input_img = input_img.to(device)
        gt_img = gt_img.to(device)
        output_teacher = teacher(input_img)
        output_student = student(input_img)

        loss = mse_loss(output_student, output_teacher)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / total_batches
    return average_loss

def eval_student(student, teacher, eval_loader, device):
    student.eval()
    total_loss = 0.0
    total_batches = len(eval_loader)

    with torch.no_grad():
        for input_img, gt_img in eval_loader:
            input_img = input_img.to(device)
            gt_img = gt_img.to(device)
            output_teacher = teacher(input_img)
            output_student = student(input_img)

            loss = mse_loss(output_student, output_teacher)
            total_loss += loss.item()

    average_loss = total_loss / total_batches
    return average_loss

date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


if not os.path.exists(save_folder):
        os.makedirs(save_folder)

def save_model(model, epoch):
    path = f"{save_folder}/student_model_epoch_{epoch}.pth"
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

optimizer = torch.optim.Adam(student.parameters(), lr=2e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
num_of_epochs = 20

train_losses = []
eval_losses = []
t = []
best_loss = None
epochs_no_improve = 0
patience = 5
min_delta = 0.001
for epoch in range(num_of_epochs):
	t_inicial = time.time()
	train_loss = train_student(student, teacher, optimizer, training_data_loader, device)
	eval_loss = eval_student(student, teacher, testing_data_loader, device)
	scheduler.step()


	train_losses.append(train_loss)
	eval_losses.append(eval_loss)
	print(f'Epoch {epoch+1}, Training Loss: {train_loss}, Evaluation Loss: {eval_loss}')
	save_model(student, epoch+1)
	t_pred= time.time()-t_inicial
	t.append(t_pred)
	print_log(epoch+1, num_of_epochs, t_pred, date_time ) 
	if best_loss is None or eval_loss < best_loss - min_delta:
		best_loss = eval_loss
		epochs_no_improve = 0
	else:
		epochs_no_improve += 1
	if epochs_no_improve >= patience:
		print("Early stopping triggered")
		break

# Graficar los resultados
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(eval_losses, label='Evaluation Loss')
plt.title('Training and Evaluation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{save_folder}/Loss_{date_time}.png')
plt.show()


