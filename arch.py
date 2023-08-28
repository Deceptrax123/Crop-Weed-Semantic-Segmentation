import torch 
from torch import nn 
import torchvision 
from torchvision.models import VGG16_Weights,vgg16
from layer_blocks import Self_embedding_block,Reconsructor
from unet import Unet
from torch.nn import Module
from torch.nn import Conv2d,ConvTranspose2d
from torchsummary import summary

#initialize feature extractor VGG16
feature_extractor=vgg16(weights=VGG16_Weights.DEFAULT)
classifier_1=nn.Sequential(*list(feature_extractor.classifier.children())[:-7])

feature_extractor.classifier=classifier_1

classifier_2=nn.Sequential(*list(feature_extractor.features._modules.values())[:-1])
feature_extractor.classifier=classifier_2

final_extractor=feature_extractor.classifier


#Architecture
class Architecture(Module):
    def __init__(self):
        super(Architecture,self).__init__()

        self.emb=Self_embedding_block()
        self.unet=Unet()

        self.dconv=ConvTranspose2d(in_channels=512,out_channels=256,stride=1,padding=1,kernel_size=(3,3))

        self.recon=Reconsructor()
    
    def forward(self,x):

        features=final_extractor(x)
        x=self.emb(x)

        f_comb=torch.concat((features,x))

        x=self.dconv(f_comb)
        x=self.unet(x)
        x=self.recon(x)

        return x
