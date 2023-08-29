import torch 
from torch import nn 
import torchvision 
from torchvision.models import VGG16_Weights,vgg16
from layer_blocks import Self_embedding_block,Reconsructor
from unet import Unet
from torch.nn import Module
from torch.nn import Conv2d,ConvTranspose2d
from torchsummary import summary
from vgg16 import extractor

#Architecture
class Architecture(Module):
    def __init__(self):
        super(Architecture,self).__init__()

        self.emb=Self_embedding_block()
        self.unet=Unet()

        self.dconv=ConvTranspose2d(in_channels=512,out_channels=256,stride=1,padding=1,kernel_size=(3,3))

        self.recon=Reconsructor()
    
    def forward(self,x):

        #features=extractor().classifier(x)
        emb=self.emb(x)

        #f=torch.add(features,emb,alpha=1)

        x=self.dconv(emb)
        x=self.unet(x)
        x=self.recon(x)

        return x