import torch 
from torch.nn import Conv2d,BatchNorm2d,ReLU,Dropout2d,ConvTranspose2d
from torch.nn import Module 

class Cnn_Block_extractor(Module):
    def __init__(self,features):
        super(Cnn_Block_extractor,self).__init__()

        self.conv=Conv2d(in_channels=features,out_channels=features*2,padding=1,kernel_size=(3,3),stride=1)
        self.bn=BatchNorm2d(features*2)
        self.dp=Dropout2d()
        self.relu=ReLU()
     
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.dp(x)
        x=self.relu(x)

        return x 

class Cnn_Block_recon(Module):
    def __init__(self,features):
        super(Cnn_Block_recon,self).__init__()

        self.conv=ConvTranspose2d(in_channels=features,out_channels=features//2,padding=1,kernel_size=(3,3))
        self.bn=BatchNorm2d(features//2)
        self.dp=Dropout2d()
        self.relu=ReLU()

    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.dp(x)
        x=self.relu(x)

        return x