import argparse
import os
import torch
import numpy as np
import warnings
import cv2
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from Student_transweather_model import Student_Transweather  # Assuming this is your custom model
from transweather_model import Transweather
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='video_niebla.mp4', help='nombre del video + .mp4')
parser.add_argument('--model', type=int, default=0, help='0 original, 1 KD, 2 Light_model, 3 encoder, 4 decoder')
opt = parser.parse_args()

device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# --- Load Networks --- #
if opt.model == 0:  # Original
    model_path = 'best_model'
    net = Transweather()
elif opt.model == 1:  # KD
    model_path = 'Best_KD'
    net = Student_Transweather()
elif opt.model == 2:
    model_path = 'Best_LightModel'
    net = Student_Transweather()   
elif opt.model == 3:
    model_path = 'Best_Encoder.pth'
    net = Student_Transweather()
else:
    model_path = 'Best_Decoder.pth'
    net = Student_Transweather()
net = nn.DataParallel(net, device_ids=device_ids)
net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
net.eval() 

# Initialize video capture
vid = cv2.VideoCapture(f"../videos/{opt.video}")
width = 640
height = 480
cv2.namedWindow("video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("video", width, height)

while vid.isOpened():
    ret, frame = vid.read()
    if ret:
        #frame = cv2.resize(frame, (width, height))  # Resize frame if necessary
        cv2.imshow('video', frame)
        cv2.waitKey(1)

        # Convert frame to RGB PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Apply transformations
        fixed_size = (256, 256)
        pil_image = pil_image.resize(fixed_size, Image.LANCZOS)
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        tensor_image = transform_input(pil_image).to(device)
        with torch.no_grad():
            t = tensor_image.unsqueeze(0).to(device)
            dehaze, _, _ = net(t)
            
            # Convert tensor back to numpy array for OpenCV processing
            frame_processed = dehaze.squeeze(0).cpu().permute(1, 2, 0).numpy()
            frame_processed = (frame_processed * 255).astype(np.uint8)
            frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_RGB2BGR)

        cv2.imshow("processed", frame_processed)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

vid.release()
cv2.destroyAllWindows()
