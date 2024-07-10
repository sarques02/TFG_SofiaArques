from dataloader_udr import *
# from UDR_S2Former import *

from Student_UDR import *
import torch
from metrics import *
from psnr_ssim import *
from loss.CL1 import L1_Charbonnier_loss, PSNRLoss 
from loss.perceptual import PerceptualLoss2
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import datetime

# CPU or GPU
import torch

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
    print("CUDA (GPU) is available.")
else:
    device = torch.device("cpu")   # Use CPU
    print("CUDA (GPU) is not available. Switching to CPU.")



# Dataloaders
crop_size = 128

train_set = RainDS_Dataset("RainDS/RainDS_syn/",train=True,crop=True,size=crop_size)
val_set = RainDS_Dataset("RainDS/RainDS_syn/",train=False,crop=True,size=crop_size)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=2, shuffle=False, num_workers=4)

# Model
model = Transformer_Student(img_size=(crop_size, crop_size))
model.to(device)

# Training parameters
epochs = 50
initlr = 0.0007
optimizer = torch.optim.Adam(model.parameters(), lr=initlr, betas=[0.9,0.999])
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=initlr,max_lr=1.2*initlr,step_size_up=400,cycle_momentum=False)
loss_fO = PSNRLoss()
loss_l1 = nn.L1Loss()
loss_perO = PerceptualLoss2()

loss_total_train = []
loss_total_val = []

date = datetime.datetime.now().strftime('%d-%m-%y-%H_%M')
save_folder = 'OnlyStudent/{}/'.format(date)
if not os.path.exists(save_folder):
        os.makedirs(save_folder)

def print_log(epoch, num_epochs, one_epoch_time, date, loss):
    one_epoch_time = float(one_epoch_time)
    # --- Write the training log --- #
    with open(f"{save_folder}/OnlyStudent_log_{date}.txt", 'a') as f:
        print('Epoch: [{2}/{3}], Date: {0}s, Time_Cost: {1:.0f}s, Loss: {4}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, loss), file=f)


mse_loss = torch.nn.MSELoss()

def train_student(student, optimizer, train_loader, device):
    print("Train student")
    student.train()
    total_loss = 0.0
    total_batches = len(train_loader)
    
    for batch in train_loader:
        x, y, _ = batch
        x, y = x.to(device), y.to(device)
        y_list_student, var_list_student = student(x)

        loss = mse_loss(y, y_list_student[0])
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / total_batches
    return average_loss


def eval_student(student, eval_loader, device):
    print("Eval student")
    student.eval()
    total_loss = 0.0
    total_batches = len(eval_loader)

    with torch.no_grad():
        for batch in eval_loader:
            x, y, _ = batch
            x, y = x.to(device), y.to(device)
            y_list_student, var_list_student = student(x)

            loss = mse_loss(y, y_list_student[0])
            total_loss += loss.item()

    average_loss = total_loss / total_batches
    return average_loss

timestamp = str(time.time())

best_loss = 0.0
epochs_no_improve = 0
patience = 7
min_delta =  0.001


train_losses = []
eval_losses = []
t=[]

for epoch in tqdm(range(epochs)):
    t_inicial = time.time()
    train_loss = train_student(model, optimizer, train_loader, device)
    eval_loss = eval_student(model, val_loader, device)
    scheduler.step()

    train_losses.append(train_loss)
    eval_losses.append(eval_loss)

    t_pred= time.time()-t_inicial
    t.append(t_pred)
    print_log(epoch, epochs, t_pred, date, eval_loss)
    torch.save(model.state_dict(), '{}/epoch{}_loss{}.pth'.format(save_folder, epoch, eval_loss))

    if best_loss is None or eval_loss < best_loss - min_delta:
        best_loss = eval_loss
        epochs_no_improve = 0
        
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

# eval_losses = [loss.cpu().item() for loss in eval_losses]
# train_losses = [loss.cpu().item() for loss in train_losses]

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training loss')
plt.plot(eval_losses, label='Evaluation loss')
plt.title(f'Training and Evaluation loss for Student')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"{save_folder}/OnlyStudent_{date}.png")
plt.show()