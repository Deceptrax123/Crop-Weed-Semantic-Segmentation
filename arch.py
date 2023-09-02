import torch 
from torch import nn 
import torchvision 
from torchvision.models import VGG16_Weights,vgg16
from layer_blocks import Self_embedding_block,Reconsructor
from unet import Unet
from torch.nn import Module
from torch.nn import Conv2d,ConvTranspose2d
from torchsummary import summary

#Architecture
class Architecture(Module):
    def __init__(self):
        super(Architecture,self).__init__()

        self.emb=Self_embedding_block()
        self.unet=Unet()
        self.recon=Reconsructor()
    
    def forward(self,x):
        x=self.emb(x)
        x=self.unet(x)
        x=self.recon(x)

        return x

#model=Architecture()
#summary(model,input_size=(3,1024,1024),batch_size=8,device='cpu')