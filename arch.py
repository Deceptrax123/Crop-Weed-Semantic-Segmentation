import torch 
from torch import nn 
from torch.nn import Conv2d,ConvTranspose2d,ReLU,MaxPool2d,Module 
import torchvision 
from torchvision.models import vgg16_bn
from layer_blocks import Unet_block,Self_embedding_block
from torchsummary import summary


#feature extractor
model=vgg16_bn(pretrained=True)
new_classifier=nn.Sequential(*list(model.classifier.children())[:-7])
model.classifier=new_classifier



summary(model,input_size=(3,1023,1024),batch_size=8,device='cpu')