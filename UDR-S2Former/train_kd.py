import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataloader_udr import *
import time

from Teacher_UDR import Transformer
from Student_UDR import Transformer_Student


batch_size = 8
crop_size = 128
train_set = RainDS_Dataset("RainDS/RainDS_syn", train=True, crop=True, size=crop_size)
training_data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_set = RainDS_Dataset("RainDS/RainDS_syn", train=False, crop=True, size=crop_size)
testing_data_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

teacher = Transformer(img_size=(crop_size, crop_size))
teacher.load_state_dict(torch.load('pretrained/udrs2former_raindrop_syn.pth', map_location=device))
teacher.eval().to(device)

student = Transformer_Student(img_size=(crop_size, crop_size))
student.train().to(device)

date = datetime.datetime.now().strftime('%d-%m-%y-%H_%M')
save_folder = 'KD_Student/{}/'.format(date)
if not os.path.exists(save_folder):
        os.makedirs(save_folder)

def print_log(epoch, num_epochs, one_epoch_time, date, loss):
    one_epoch_time = float(one_epoch_time)
    # --- Write the training log --- #
    with open(f"{save_folder}/KD_Student_log_{date}.txt", 'a') as f:
        print('Epoch: [{2}/{3}], Date: {0}s, Time_Cost: {1:.0f}s, Loss: {4}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, loss), file=f)


def train_student(student, teacher, optimizer, train_loader, device):
    print("Train student")
    student.train()
    total_loss = 0.0
    total_batches = len(train_loader)
    
    for batch in train_loader:
        x, y, _ = batch
        x, y = x.to(device), y.to(device)
        y_list_teacher, var_list_teacher = teacher(x)
        y_list_student, var_list_student = student(x)

        loss = mse_loss(y_list_teacher[0], y_list_student[0]) + mse_loss(y_list_teacher[1], y_list_student[1]) + mse_loss(y_list_teacher[2], y_list_student[2]) + mse_loss(y_list_teacher[3], y_list_student[3]) + mse_loss(y_list_teacher[4], y_list_student[4])
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / total_batches
    return average_loss


def eval_student(student, teacher, eval_loader, device):
    print("Eval student")
    student.eval()
    total_loss = 0.0
    total_batches = len(eval_loader)

    with torch.no_grad():
        for batch in eval_loader:
            x, y, _ = batch
            x = x.to(device)
            y_list_teacher, var_list_teacher = teacher(x)
            y_list_student, var_list_student = student(x)

            loss = mse_loss(y_list_teacher[0], y_list_student[0]) + mse_loss(y_list_teacher[1], y_list_student[1]) + mse_loss(y_list_teacher[2], y_list_student[2]) + mse_loss(y_list_teacher[3], y_list_student[3]) + mse_loss(y_list_teacher[4], y_list_student[4])
            total_loss += loss.item()

    average_loss = total_loss / total_batches
    return average_loss


initlr = 0.0007
optimizer = torch.optim.Adam(student.parameters(), lr=initlr, betas=[0.9,0.999])
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=initlr, max_lr=1.2*initlr, step_size_up=400, cycle_momentum=False)
#optimizer = torch.optim.Adam(student.parameters(), lr=2e-4)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

mse_loss = torch.nn.MSELoss()
num_of_epochs = 30

best_loss = 0.0
epochs_no_improve = 0
patience = 10
min_delta =  0.001

train_losses = []
eval_losses = []
t = []

for epoch in tqdm(range(num_of_epochs)):
    t_inicial = time.time()
    train_loss = train_student(student, teacher, optimizer, training_data_loader, device)
    eval_loss = eval_student(student, teacher, testing_data_loader, device)
    print(f"Train Loss: {train_loss}")
    print(f"Eval Loss: {eval_loss}")
    scheduler.step()

    train_losses.append(train_loss)
    eval_losses.append(eval_loss)
    t_pred= time.time()-t_inicial
    t.append(t_pred)
    print_log(epoch, num_of_epochs, t_pred, date, eval_loss)
    torch.save(student.state_dict(), '{}/epoch{}_loss{}.pth'.format(save_folder, epoch, eval_loss))

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
plt.savefig(f"KD_Student_{date}.png")
plt.show()
