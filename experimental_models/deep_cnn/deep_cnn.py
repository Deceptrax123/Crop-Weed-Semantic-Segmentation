import torch 
from torch.nn import Conv2d,Dropout2d,ReLU,ConvTranspose2d
from torch.nn import Module 
from cnn_block import Cnn_Block_extractor,Cnn_Block_recon
from torchsummary import summary


class Deep_CNN(Module):
    def __init__(self):
        super(Deep_CNN,self).__init__()

        self.conv=Conv2d(in_channels=3,out_channels=4,kernel_size=(3,3),padding=1,stride=1)
        
        #Custom blocks
        self.block1=Cnn_Block_extractor(4)
        self.block2=Cnn_Block_extractor(8)
        self.block3=Cnn_Block_extractor(16)
        self.block4=Cnn_Block_extractor(32)

        self.block5=Cnn_Block_recon(64)
        self.block6=Cnn_Block_recon(32)
        self.block7=Cnn_Block_recon(16)
        self.block8=Cnn_Block_recon(8)

        self.dconv=ConvTranspose2d(in_channels=4,out_channels=3,kernel_size=(3,3),padding=1,stride=1)
    
    def forward(self,x):
        x=self.conv(x)
        
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)

        x=self.block5(x)
        x=self.block6(x)
        x=self.block7(x)
        x=self.block8(x)

        x=self.dconv(x)

        return x

model=Deep_CNN()

summary(model,input_size=(3,1024,1024),batch_size=8,device='cpu')