### ----------------- IMPORTS ----------------- ###
import torch.onnx
import torch
import torch.nn.utils.prune as prune
from tqdm import tqdm
import onnx
from onnxconverter_common import float16
import nncf
from nncf import compress_weights
from torch.utils.data import DataLoader
import openvino as ov
from sklearn.metrics import accuracy_score
import torch.nn as nn
import time
from typing import List, Optional
import re
import subprocess
from collections import OrderedDict

import onnx
from neural_compressor.experimental import Quantization, common
from nncf import compress_weights


### ---------------- VARIABLES ---------------- ###

# Imports específicos de la red a utilizar
from transweather_model import *
from val_data_functions import ValData
## -----
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nombre = "Transweather"
model = Transweather().to(device)
pesos = 'best'

dummy_input = torch.rand((1, 3, 432, 640)).to(device)

val_filename1 = './allweather/allweather.txt'
val_data_dir = ''
val_data_loader = DataLoader(ValData(val_data_dir,val_filename1), batch_size=1, shuffle=False, num_workers=8)
calibration_loader = DataLoader(ValData(val_data_dir,val_filename1), batch_size=1, shuffle=False, num_workers=8)




### -------- TRANSFORMAR A MODELO ONNX -------- ###
def to_onnx(modelo = model, nombre = nombre):

    state_dict = torch.load(pesos, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Eliminar `module.`
        else:
            new_state_dict[k] = v
    modelo = Transweather()

    modelo.load_state_dict(new_state_dict)
    modelo.to(device)

    ### Export ###
    torch.onnx.export(modelo.to(device),                    # model being run 
        dummy_input.to(device),                             # model input (or a tuple for multiple inputs) 
        f"{nombre}_og.onnx",                                # where to save the model
        export_params=True,                                 # store the trained parameter weights inside the model file 
        opset_version=16,                                   # the ONNX version to export the model to 
        do_constant_folding=False,                          # whether to execute constant folding for optimization 
        input_names = ['modelInput'],                       # the model's input names 
        output_names = ['modelOutput'],                     # the model's output names 
        dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                    'modelOutput' : {0 : 'batch_size'}}) 

    print(" ") 
    print('Model has been converted to ONNX')


### ------------ TRANSFORMAR a FP16 ----------- ###
def to_fp16(nombre = nombre):
    model = onnx.load(f"{nombre}_og.onnx")
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, f"{nombre}_fp16.onnx")
    print('Model has been quantized to fp16')

### ------------ TRANSFORMAR a INT8 ----------- ###

def to_int8_metodo2():
    model_path = f"{nombre}_og.onnx"
    quantize_static(model_fp32, model_quant, calibration_dataset)


to_onnx()
to_fp16()
to_int8()
