import torch 
from torch import nn

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.kaiming_normal_(m.weight.data,mode='fan_in',nonlinearity='relu')
            