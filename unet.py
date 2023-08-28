import torch 
from torch import nn  
from layer_blocks import Unet_decoding_block,Unet_encoding_block
from torch.nn import Module,ConvTranspose2d,Conv2d
from torchsummary import summary

class Unet(Module):
    def __init__(self):
        super(Unet,self).__init__()

        #downsampling blocks
        self.down1=Unet_encoding_block(256)
        self.down2=Unet_encoding_block(512)
        self.down3=Unet_encoding_block(1024)

        #embedding convolution
        self.emb=Conv2d(in_channels=2048,out_channels=2048,stride=1,padding=1,kernel_size=(3,3))
        
        #upsampling blocks
        self.up1=Unet_decoding_block(2048)
        self.up2=Unet_decoding_block(1024)
        self.up3=Unet_decoding_block(512)

    def forward(self,x):
        x=self.down1(x)
        x=self.down2(x)
        x=self.down3(x)

        x=self.emb(x)

        x=self.up1(x)
        x=self.up2(x)
        x=self.up3(x)

        return x 

#model=Unet()
#summary(model,input_size=(256,32,32),batch_size=8,device='cpu')
