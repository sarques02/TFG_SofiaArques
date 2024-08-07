# --- Imports --- #
import time
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10
from skimage import metrics
import pdb
import os
from tqdm import tqdm

def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)
    # import ipdb  
    # ipdb.set_trace()
    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    # print(dehaze_list_np[0].shape, gt_list_np[0].shape)
    ssim_list = [metrics.structural_similarity(dehaze_list_np[ind],  gt_list_np[ind], win_size=11, channel_axis=2, data_range=1) for ind in range(len(dehaze_list))]

    return ssim_list
def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min)

    return img
def norm_range(t, range):
    if range is not None:
        return norm_ip(t, range[0], range[1])
    else:
        return norm_ip(t, t.min(), t.max())
    #return norm_ip(t, t.min(), t.max())

def validation(net, val_data_loader, device, save_tag=False):
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
            save_image(dehaze, image_name)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim


def save_image(dehaze, image_name):
    dehaze_images = torch.split(dehaze, 1, dim=0)
    batch_num = len(dehaze_images)

    results_dir = './results/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    
    for ind in range(batch_num):
        utils.save_image(dehaze_images[ind], './results/prueba31/{}'.format(os.path.basename(image_name[ind])[:-3] + 'png'))


def print_log(epoch, num_epochs, one_epoch_time, train_psnr, train_ssim, val_psnr, val_ssim, log_file, date):
    one_epoch_time = float(one_epoch_time)
    train_psnr = float(train_psnr)
    train_ssim = float(train_ssim)
    val_psnr = float(val_psnr)
    val_ssim = float(val_ssim)
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Train_PSNR:{6:.2f}, Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim, train_ssim))

    # --- Write the training log --- #
    with open(f"log_{date}.txt", 'a') as f:
        print('Epoch: [{2}/{3}], Date: {0}s, Time_Cost: {1:.0f}s,  Train_PSNR: {4:.2f}, Train_SSIM: {7:.2f}, Val_PSNR: {5:.2f}, Val_SSIM: {6:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim, train_ssim), file=f)


def adjust_learning_rate(optimizer, epoch, lr_decay=0.5):

    # --- Decay learning rate --- #
    # step = 100 if category == 'indoor' else 30
    step = 30

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))
