import glob
import torch
import os
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np

import torch.nn as nn
from metrics_kd import dice_loss
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import datetime
import time

# from student_network_restoration import *
from student_network_haze1 import *
# from student_network_rain2 import *
# from student_network_snow3 import *

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

batch_size = 64

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

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_of_epochs = 20
fecha = datetime.datetime.now().strftime('%d-%m-%y-%H_%M')
net_num = 1 #0 restoration, 1 haze, 2 rain, 3 snow
factor_num= 4 #4,8
save_folder = 'OnlyStudent/net{}/factor{}/{}'.format(net_num, factor_num, fecha)
if not os.path.exists(save_folder):
        os.makedirs(save_folder)

def print_log(epoch, num_epochs, one_epoch_time, date, loss):
    one_epoch_time = float(one_epoch_time)
    # --- Write the training log --- #
    with open(f"{save_folder}/OnlyStudent_log_{date}.txt", 'a') as f:
        print('Epoch: [{2}/{3}], Date: {0}s, Time_Cost: {1:.0f}s, Loss: {4}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, loss), file=f)


def evaluate(student, eval_loader, device):
    student.eval()

    criterion = nn.BCEWithLogitsLoss()
    ll = []
    with torch.no_grad():
        for i,(img,gt) in tqdm(enumerate(eval_loader)):
            if torch.cuda.is_available():
                img, gt = img.to(device), gt.to(device)
            img, gt = Variable(img), Variable(gt)

            output = student(img)
            output = output.clamp(min = 0, max = 1)
            gt = gt.clamp(min = 0, max = 1)
            loss = dice_loss(output, gt)
            ll.append(loss.item())



    mean_dice = np.mean(ll)
    print('Eval metrics:\n\tAverabe Dice loss:{}'.format(mean_dice))
    return mean_dice


def train(student, optimizer, train_loader, device):
    print(' --- student training')
    student.train().cuda()
    criterion = nn.BCEWithLogitsLoss()
    ll = []
    for i, (img, gt) in tqdm(enumerate(train_loader)):
        # print('i', i)
        if torch.cuda.is_available():
            img, gt = img.to(device), gt.to(device)

            img, gt = Variable(img), Variable(gt)
            output = student(img)
            output = output.clamp(min = 0, max = 1)
            gt = gt.clamp(min = 0, max = 1)
            loss = dice_loss(output, gt)
            ll.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

    mean_dice = np.mean(ll)
    print("Average loss over this epoch:\n\tDice:{}".format(mean_dice))
    return mean_dice

if __name__ == "__main__":

    student = My_net()
    # student = restore_net()
    student = student.to(device)

    #Optimizador y Scheduler
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3) 
    scheduler = StepLR(optimizer, step_size = 100, gamma = 0.2)

    best_loss = 0.0
    epochs_no_improve = 0
    patience = 5
    min_delta =  0.001
    
    train_losses = []
    eval_losses = []
    t = []
    for epoch in tqdm(range(num_of_epochs)):
        t_inicial = time.time()
        print(' --- student training: epoch {}'.format(epoch+1))
        
        train_loss = train(student, optimizer, training_data_loader, device)
        eval_loss = evaluate(student, testing_data_loader, device)

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)

        #if val_metric is best, add checkpoint
        torch.save(student.state_dict(), '{}/epoch{}_loss{}.pth'.format(save_folder, epoch, eval_loss))
        print("Checkpoint {} saved!".format(epoch+1))
        scheduler.step()
        t_pred= time.time()-t_inicial
        t.append(t_pred)
        print_log(epoch+1, num_of_epochs, t_pred, fecha, eval_loss)

        if best_loss is None or eval_loss < best_loss - min_delta:
            best_loss = eval_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break
        


# eval_losses = [loss.cpu().item() for loss in eval_losses]
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training loss')
plt.plot(eval_losses, label='Evaluation loss')
plt.title(f'Training and Evaluation loss for net {net_num} with factor {factor_num}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"{save_folder}/OnlyStudent_{fecha}.png")
plt.show()
