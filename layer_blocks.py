import torch 
from torch.nn import Conv2d,ConvTranspose2d,ReLU,BatchNorm2d,MaxPool2d,AvgPool2d
from torch.nn import Module 
from torchsummary import summary

class Unet_block(Module):
    def __init__(self,features):
        super(Unet_block,self).__init__()

        self.conv1=Conv2d(in_channels=features,out_channels=2*features,stride=1,kernel_size=(3,3))
        self.bn1=BatchNorm2d(features)
        self.relu1=ReLU()

        self.conv2=Conv2d(in_channels=2*features,out_channels=2*features,stride=1,kernel_size=(3,3))
        self.bn2=BatchNorm2d(features)
        self.relu2=ReLU()

        self.maxpool=MaxPool2d(kernel_size=(2,2))

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu1(x)

        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu2(x)

        x=self.maxpool(x)

        return x

class Self_embedding(Module):
    def __init__(self):
        super(Self_embedding,self).__init__()

        self.bconv1=Conv2d(in_channels=3,out_channels=4,padding=1,stride=2)
        self.bn1=BatchNorm2d()
        self.relu1=ReLU()

        self.bconv2=Conv2d(in_channels=4,out_channels=8,stride=2,padding=1)
        self.bn2=BatchNorm2d()
        self.relu2=ReLU()

        self.bconv3=Conv2d(in_channels=8,out_channels=16,stride=2,padding=1)
        self.bn3=BatchNorm2d()
        self.relu3=ReLU()

        self.bconv4=Conv2d(in_channels=16,out_channels=32,stride=2,padding=1)
        self.bn4=BatchNorm2d()
        self.relu4=ReLU()

    def forward(self,x):
        x=self.bconv1(x)
        x=self.bn1(x)
        x=self.relu1(x)

        x=self.bconv2(x)
        x=self.bn2(x)
        x=self.relu2(x)

        x=self.bconv3(x)
        x=self.bn3(x)
        x=self.relu3(x)

        x=self.bconv4(x)
        x=self.bn4(x)
        x=self.relu4(x)

        return x