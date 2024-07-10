import argparse
import os
from os.path import exists, join as join_paths
import torch
import numpy as np
import warnings
import cv2
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from PIL import Image
from metrics import *
import network_model1 as network_haze1
import network_model2 as network_rain2
import network_model3 as network_snow3
import time
import torchvision.transforms as tfs 

'''
Pulsar 'q' para cerrar videos.
Si el código no da errores pero no saca el vídeo comprobar nombre del video y su ruta.
'''

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='video_niebla.mp4', help='nombre del video + .mp4')
parser.add_argument('--models', type=int, default=0, help='0 original, 1 KD, 2 Light_model')
parser.add_argument('--factor', default= 4, type=int)

opt = parser.parse_args()
factor_value = opt.factor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

width = 640  # Anchura de la ventana
height = 480  # Altura de la ventana

if opt.models == 0:
    print("Usando modelos originales")
    d1_path = f"./checkpoints_originales/domain1.pth" #Path a los pesos
    d2_path = f"./checkpoints_originales/domain2.pth"
    d3_path = f"./checkpoints_originales/domain3.pth"
    factor_value = 1 #Factor = 1 para las redes originales
    
elif opt.models == 1:
    print("Usando modelos KD")
    d1_path = f"./KD_models/net1factor{factor_value}.pth"
    d2_path = f"./KD_models/net2factor{factor_value}.pth"
    d3_path = f"./KD_models/net3factor{factor_value}.pth"

else:
    print("Usando modelos ligeros")
    d1_path = f"./Light_models/net1factor{factor_value}.pth"
    d2_path = f"./Light_models/net2factor{factor_value}.pth"
    d3_path = f"./Light_models/net3factor{factor_value}.pth"



## Load Networks ##
G_path = "./checkpoints_originales/netG_model.pth" #Como no se comprime la red de restauración siempre será este path
my_net = torch.load(G_path)
my_net.to(device)
my_net.eval()


d1_net = network_haze1.My_net(factor = factor_value)
#si es el original carga directa, si es kd o student carga state dict
if opt.models == 0:
    d1_net = torch.load(d1_path)
else:
    d1_net.load_state_dict(torch.load(d1_path, map_location=device))
d1_net.to(device)
d1_net.eval()

d2_net = network_rain2.My_net(factor = factor_value)
if opt.models == 0:
    d2_net = torch.load(d2_path)
else:
    d2_net.load_state_dict(torch.load(d2_path, map_location=device))
d2_net.to(device)
d2_net.eval()

d3_net = network_snow3.My_net(factor = factor_value)
if opt.models == 0:
    d3_net = torch.load(d3_path)
else:
    d3_net.load_state_dict(torch.load(d3_path, map_location=device))

d3_net.to(device)
d3_net.eval()

# Define the transformation pipeline
transform = tfs.Compose([
    tfs.Resize((256, 256), interpolation=Image.BICUBIC),
    tfs.ToTensor(),
    tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Inicia la captura del video
vid = cv2.VideoCapture(f"../videos/{opt.video}")

cv2.namedWindow("video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("video", width, height)

while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        break

    # Resize frame to 640x480
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow('video', frame)

    # Convert frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    width = frame.shape[0]
    a = frame[:,:width*1, :]
    a1 = cv2.resize(a, (256, 256), interpolation=cv2.INTER_CUBIC)
    a1 = tfs.ToTensor()(a1)    
    tensor_image = tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a1)

    with torch.no_grad():
        haze = tensor_image.unsqueeze(0).to(device)
        fake_d1 = d1_net(haze)
        fake_d2 = d2_net(haze)
        fake_d3 = d3_net(haze)
        dehaze = my_net(haze, fake_d1, fake_d2, fake_d3)

        frame_processed = dehaze.cpu().squeeze().permute(1, 2, 0).numpy()
        
        # Unnormalize the image
        frame_processed = (frame_processed * 0.5 + 0.5) * 255
        frame_processed = frame_processed.astype(np.uint8)
        
        frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_RGB2BGR)
        frame_processed = cv2.resize(frame_processed, (frame.shape[1], frame.shape[0]))
        
    cv2.imshow("processed", frame_processed)
    if cv2.waitKey(25) & 0xFF == ord('q'):
            break


vid.release()
cv2.destroyAllWindows()
