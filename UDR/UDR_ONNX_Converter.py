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

### ---------------- VARIABLES ---------------- ###

# Imports específicos de la red a utilizar
from Student_UDR import *
from dataloader_udr import *

## -----


nombre = "udr_real"
model = Transformer_Student(img_size=(320, 320)).cpu()
pesos = 'UDR_BEST_LIGHT.pth'

dummy_input = torch.rand((1,3,320,320))

path_to_dataset = '/home/tft5/UDR-S2Former_deraining-main/datasets/Reducido_10'
#calibration_loader = DataLoader(dataset=RainDS_Dataset('datasets/RainDS/RainDS_real/transformed',train=False,dataset_type='rsrd'),batch_size=1,shuffle=False,num_workers=4)




### -------- TRANSFORMAR A MODELO ONNX -------- ###
def to_onnx(modelo = model, nombre = nombre):
    ckpt = torch.load(pesos, map_location=torch.device('cpu'))
    modelo.load_state_dict(ckpt)

    ### Export ###
    torch.onnx.export(modelo,         # model being run 
        dummy_input,       # model input (or a tuple for multiple inputs) 
        f"{nombre}.onnx",       # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file 
        opset_version=16,    # the ONNX version to export the model to 
        do_constant_folding=True,  # whether to execute constant folding for optimization 
        input_names = ['modelInput'],   # the model's input names 
        output_names = ['modelOutput'], # the model's output names 
        dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                    'modelOutput' : {0 : 'batch_size'}}) 

    print(" ") 
    print('Model has been converted to ONNX')


### ------------ TRANSFORMAR a FP16 ----------- ###
def to_fp16(nombre = nombre):
    model = onnx.load(f"{nombre}.onnx")
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, f"{nombre}_fp16.onnx")
    print('Model has been quantized to fp16')
### ------------ TRANSFORMAR a INT8 ----------- ###

'''
def to_int8(calibration = calibration_loader, nombre = nombre):
    model_onnx = onnx.load(f"{nombre}.onnx")
    
    def transform_fn(data_item):
        images, clear, name = data_item
        return {'modelInput': images.numpy()}

    #Si transform_fn falla, ejecutar esto para obtener el nombre de los inputs, sustituir en modelInput (del return):
    # print([input.name for input in model_onnx.graph.input]) 

    calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)
    onnx_quantized_model = nncf.quantize(model_onnx, calibration_dataset, subset_size=30)

    onnx.save_model(onnx_quantized_model, f"{nombre}_int8.onnx")
    print('Model has been quantized to int8')

'''
import onnx
from neural_compressor.experimental import Quantization, common
from nncf import compress_weights

# def to_int8(model_path):
#     model = onnx.load('udr.onnx')
#     quantizer = Quantization('./conf.yaml')
#     quantizer.model = common.Model(model)
#     q_model = quantizer()
#     q_model.save('./outputs/')


import onnx
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
def to_int8():
    model_path = f"{nombre}.onnx"
    # def transform_fn(data_item):
    #         images, clear, name = data_item
    #         return {'modelInput': images.numpy()}

    # #Si transform_fn falla, ejecutar esto para obtener el nombre de los inputs, sustituir en modelInput (del return):
    # # print([input.name for input in model_onnx.graph.input]) 

    # calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)
    # model_fp32 = 'udr.onnx'
    # model_quant = 'udr_quant.onnx'
    # # quantized_model = quantize_dynamic(model_fp32, model_quant)
    # quantize_static(model_fp32, model_quant, calibration_dataset)


    quantize_dynamic(model_path, f"{nombre}_int8.onnx", weight_type=QuantType.QUInt8)
    # quantize_dynamic(model_path, model_path+".int8.quant", weight_type=QuantType.QInt8, nodes_to_quantize=linear_names, extra_options={"MatMulConstBOnly":True})
    # quantize_dynamic(model_path, model_path+".uint8.quant", weight_type=QuantType.QUInt8, nodes_to_quantize=linear_names, extra_options={"MatMulConstBOnly":True})


### ----------------- PRUNING ----------------- ###
'''
pruned_model = model #Copia del modelo


checkpoint = torch.load(pesos, map_location=torch.device('cpu'))


pruned_model.load_state_dict(checkpoint)
model.eval()

for name, module in pruned_model.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount = 0.3)

torch.save(pruned_model.state_dict(), f'{nombre}_pruned.pth')
print("Model pruned")

'''
to_onnx()
# to_fp16()
#to_int8()