### ----------------- IMPORTS ----------------- ###
# import torch.onpip insta.utils.prune as prune
# from tqdm import tqdm
import onnx
from onnxconverter_common import float16
import nncf
from nncf import compress_weights
# from torch.utils.data import DataLoader
# import openvino as ov
from sklearn.metrics import accuracy_score
# import torch.nn as nn
import time
from typing import List, Optional
import re
import subprocess
from onnxruntime.quantization import quantize_dynamic, QuantType
from collections import OrderedDict

from test import *

from network_restoration import *
import network_model1 as net1
import network_model2 as net2
import network_model3 as net3

# import torch.onnx
# from torchvision.ops.deform_conv import DeformConv2d
import deform_conv2d_onnx_exporter
import os

nombre = "domain"

#El input es del mismo tamaño para todos
dummy_input_1 = torch.rand((1, 3, 256, 256)).cuda()
dummy_input_2 = torch.rand((1, 3, 256, 256)).cuda()
dummy_input_3 = torch.rand((1, 3, 256, 256)).cuda()
dummy_input_4 = torch.rand((1, 3, 256, 256)).cuda()

deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()

def to_onnx(net=1):
    if net ==1:
        pesos='./checkpoints_originales/domain1.pth'
        clase = net1.My_net()
        print("Seleccionado el modelo 1")

    elif net == 2:
        pesos='./checkpoints_originales/domain1.pth'
        clase = net2.My_net()
        print("Seleccionado el modelo 2")

    else:
        pesos='./checkpoints_originales/domain1.pth'
        clase = net3.My_net()
        print("Seleccionado el modelo 3")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(pesos)
    # clase.load_state_dict(ckpt)
    torch.save(ckpt.state_dict(), f"domain{net}_gpu.pth")
    state_dict = torch.load(f"domain{net}_gpu.pth")

    new_state_dict = OrderedDict()

    number_of_prefixes_to_remove = 4
    prefix = 'module.' * number_of_prefixes_to_remove

    for k, v in state_dict.items():
        name = k
        new_state_dict[name] = v

    clase.load_state_dict(new_state_dict)
    # for k, v in clase.named_parameters():
    #     print(k, v)
    torch.onnx.export(clase.to(device),
                      dummy_input_1.to(device),
                      f"{nombre}{net}_gpu.onnx",
                      export_params=True,
                      opset_version=16,
                      do_constant_folding=True,
                      input_names=['modelInput'],
                      output_names=['modelOutput'],
                      dynamic_axes={'modelInput': {0: 'batch_size'},
                                    'modelOutput': {0: 'batch_size'}})

    print(" ")
    print('Model has been converted to ONNX')


def to_onnx_restoration():
    pesosG='/home/gatv-matteo/Desktop/tfg_sofia/Domain_Translation/checkpoints/netG_model.pth'
    modelo = restore_net()
    ckpt = torch.load(pesosG)
    torch.save(ckpt.module.state_dict(), "model_test.pth")
    state_dict = torch.load("model_test.pth")
    
    new_state_dict = OrderedDict()

    number_of_prefixes_to_remove = 4
    prefix = 'module.' * number_of_prefixes_to_remove  # 'module.module.module.module.'

    for k, v in state_dict.items():
        # Check if the key starts with the prefix and remove it
        if k.startswith(prefix):
            name = k[len(prefix):]  # remove prefix
        else:
            name = k
        new_state_dict[name] = v

    modelo.load_state_dict(new_state_dict)


    ### Export ###
    torch.onnx.export(modelo.cuda(),    # model being run 
        (dummy_input_1, dummy_input_2, dummy_input_3, dummy_input_4),  # model input (or a tuple for multiple inputs) 
        f"{nombre}_og.onnx",               # where to save the model
        export_params=True,             # store the trained parameter weights inside the model file 
        opset_version=16,               # the ONNX version to export the model to 
        do_constant_folding=True,       # whether to execute constant folding for optimization 
        input_names = ['input_1', 'input_2', 'input_3', 'input_4'],   # the model's input names 
        output_names = ['modelOutput'], # the model's output names 
        dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                    'modelOutput' : {0 : 'batch_size'}}) 

    print(" ") 
    print('Model has been converted to ONNX')

def to_fp16(numero, nombre = nombre):
    model = onnx.load(f"{nombre}{numero}.onnx")
    # os.system(f'python -m onnxruntime.quantization.preprocess --input {model} --output domain{numero}_infer_gpu.onnx')
    model_fp16 = float16.convert_float_to_float16(model)
    modeloo = onnx.load(f'{model}_infer.onnx')
    onnx.save(model_fp16, f"{nombre}{numero}_fp16.onnx")
    print('Model has been quantized to fp16')

def to_int8(net):
    model_path = f"{nombre}{net}.onnx"
    quantize_dynamic(model_path, f"{nombre}{net}_int8.onnx", weight_type=QuantType.QUInt8)



to_onnx_restoration()
# to_onnx(1)
# to_onnx(2)
# to_onnx(3)

to_fp16('')
# to_fp16(1)
# to_fp16(2)
# to_fp16(3)

to_int8('')
# to_int8(1)
# to_int8(2)
# to_int8(3)