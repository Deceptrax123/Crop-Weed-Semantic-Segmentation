import torch 
from torch import nn  
from layer_blocks import Unet_decoding_block,Unet_encoding_block
from torch.nn import Module,ConvTranspose2d,Conv2d,MaxPool2d,Dropout2d
from torchsummary import summary

class Unet(Module):
    def __init__(self):
        super(Unet,self).__init__()

        #downsampling blocks
        self.down1=Unet_encoding_block(32)
        self.down2=Unet_encoding_block(64)
        self.down3=Unet_encoding_block(128)

        #embedding convolution
        self.emb1=Conv2d(in_channels=256,out_channels=256,stride=1,padding=1,kernel_size=(3,3))
        self.emb2=Conv2d(in_channels=256,out_channels=256,stride=1,padding=1,kernel_size=(3,3))
        
        #upsampling blocks
        self.up1=Unet_decoding_block(256)
        self.up2=Unet_decoding_block(128)
        self.up3=Unet_decoding_block(64)
    

        #deconvolutoion blocks
        self.dconv1=ConvTranspose2d(in_channels=256,out_channels=256,stride=2,padding=1,output_padding=1,kernel_size=(3,3))
        self.dconv2=ConvTranspose2d(in_channels=128,out_channels=128,stride=2,padding=1,output_padding=1,kernel_size=(3,3))
        self.dconv3=ConvTranspose2d(in_channels=64,out_channels=64,stride=2,padding=1,output_padding=1,kernel_size=(3,3))

        #maxpools
        self.max1=MaxPool2d(kernel_size=(2,2),stride=2)
        self.max2=MaxPool2d(kernel_size=(2,2),stride=2)
        self.max3=MaxPool2d(kernel_size=(2,2),stride=2)

        #dropouts
        self.dp1=Dropout2d()
        self.dp2=Dropout2d()
        self.dp3=Dropout2d()
        self.dp4=Dropout2d()
        self.dp5=Dropout2d()
        self.dp6=Dropout2d()

    def forward(self,x):
        x1=self.down1(x)
        x1_max=self.max1(x1)
        x1_max=self.dp1(x1_max)

        x2=self.down2(x1_max)
        x2_max=self.max2(x2)
        x2_max=self.dp2(x2_max)

        x3=self.down3(x2_max)
        x3_max=self.max3(x3)
        x3_max=self.dp3(x3_max)

        x4=self.emb1(x3_max)
        x5=self.emb2(x4)

        x6=self.dconv1(x5)
        xcat1=torch.add(x6,x3)
        xcat1=self.dp4(xcat1)
        xu1=self.up1(xcat1)

        x7=self.dconv2(xu1)
        xcat2=torch.add(x2,x7)
        xcat2=self.dp5(xcat2)
        xu2=self.up2(xcat2)

        x8=self.dconv3(xu2)
        xcat3=torch.add(x8,x1)
        xcat3=self.dp3(xcat3)
        output=self.up3(xcat3)

        return output

#model=Unet()
#summary(model,input_size=(32,128,128),batch_size=8,device='cpu')
