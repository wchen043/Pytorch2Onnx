import os
os.environ['PYTHONPATH']='/home/weijia/Desktop/Git/DeepLabV3Plus-Pytorch'
from tqdm import tqdm
import network
# import utils
# import os
# import random
# import argparse
# import numpy as np

# from torch.utils import data
# from datasets import VOCSegmentation, Cityscapes
# from utils import ext_transforms as et
# from metrics import StreamSegMetrics

import torch
import torch.nn as nn
# from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


torch_model = network.deeplabv3_mobilenet(num_classes=21, output_stride=16)
x = torch.load('/home/weijia/Desktop/Git/trifo_model/best_deeplabv3_mobilenet_voc_os16.pth',map_location=torch.device('cpu'))
torch_model.load_state_dict(x['model_state'])

# set the model to inference mode
torch_model.eval()

# Input to the model
x = torch.randn(1, 3, 513, 513, requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "best_deeplabv3_mobilenet_voc_os16.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

import onnx

onnx_model = onnx.load("best_deeplabv3_mobilenet_voc_os16.onnx")
onnx.checker.check_model(onnx_model)