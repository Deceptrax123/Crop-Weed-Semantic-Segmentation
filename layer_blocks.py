import torch 
from torch.nn import Conv2d,ConvTranspose2d,ReLU,BatchNorm2d,MaxPool2d,MaxUnpool2d,AdaptiveAvgPool2d,Upsample,Dropout2d
from torch.nn import Module 
from torchsummary import summary

class Unet_encoding_block(Module):
    def __init__(self,features):
        super(Unet_encoding_block,self).__init__()

        self.conv1=Conv2d(in_channels=features,out_channels=2*features,stride=1,kernel_size=(3,3),padding=1)
        self.bn1=BatchNorm2d(2*features)
        self.relu1=ReLU()

        self.conv2=Conv2d(in_channels=2*features,out_channels=2*features,stride=1,kernel_size=(3,3),padding=1)
        self.bn2=BatchNorm2d(2*features)
        self.relu2=ReLU()

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu1(x)

        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu2(x)

        return x

class Unet_decoding_block(Module):
    def __init__(self,features):
        super(Unet_decoding_block,self).__init__()

        self.dconv1=ConvTranspose2d(in_channels=features,out_channels=features//2,stride=1,padding=1,kernel_size=(3,3))
        self.bn1=BatchNorm2d(features//2)
        self.relu1=ReLU()

        self.dconv2=ConvTranspose2d(in_channels=features//2,out_channels=features//2,kernel_size=(3,3),padding=1,stride=1)
        self.bn2=BatchNorm2d(features//2)
        self.relu2=ReLU()

    def forward(self,x):

        x=self.dconv1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        
        x=self.dconv2(x)
        x=self.bn2(x)
        x=self.relu2(x)


        return x

class Self_embedding_block(Module):
    def __init__(self):
        super(Self_embedding_block,self).__init__()

        self.bconv1=Conv2d(in_channels=3,out_channels=64,stride=2,kernel_size=(3,3),padding=1)
        self.bn1=BatchNorm2d(64)
        self.relu1=ReLU()
        self.dp1=Dropout2d()

        self.bconv2=Conv2d(in_channels=64,out_channels=128,stride=2,kernel_size=(3,3),padding=1)
        self.bn2=BatchNorm2d(128)
        self.relu2=ReLU()
        self.dp2=Dropout2d()

        self.bconv3=Conv2d(in_channels=128,out_channels=256,stride=2,kernel_size=(3,3),padding=1)
        self.bn3=BatchNorm2d(256)
        self.relu3=ReLU()
        self.dp3=Dropout2d()

        self.bconv4=Conv2d(in_channels=256,out_channels=512,stride=2,kernel_size=(3,3),padding=1)
        self.bn4=BatchNorm2d(512)
        self.relu4=ReLU()
        self.dp4=Dropout2d()



    def forward(self,x):
        x=self.bconv1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.dp1(x)

        x=self.bconv2(x)
        x=self.bn2(x)
        x=self.relu2(x)
        x=self.dp2(x)

        x=self.bconv3(x)
        x=self.bn3(x)
        x=self.relu3(x)
        x=self.dp3(x)

        x=self.bconv4(x)
        x=self.bn4(x)
        x=self.relu4(x)
        x=self.dp4(x)

        return x

class Reconsructor(Module):
    def __init__(self):
        super(Reconsructor,self).__init__()

        self.dconv1=ConvTranspose2d(in_channels=256,out_channels=128,stride=2,padding=1,output_padding=1,kernel_size=(3,3))
        self.bn1=BatchNorm2d(128)
        self.relu1=ReLU()
        self.dp1=Dropout2d()

        self.dconv2=ConvTranspose2d(in_channels=128,out_channels=64,padding=1,output_padding=1,kernel_size=(3,3),stride=2)
        self.bn2=BatchNorm2d(64)
        self.relu2=ReLU()
        self.dp2=Dropout2d()

        self.dconv3=ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=(3,3),padding=1,output_padding=1,stride=2)
        self.bn3=BatchNorm2d(32)
        self.relu3=ReLU()
        self.dp3=Dropout2d()

        self.dconv4=ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=(3,3),padding=1,output_padding=1,stride=2)
        self.bn4=BatchNorm2d(16)
        self.relu4=ReLU()
        self.dp4=Dropout2d()

        self.dconv5=ConvTranspose2d(in_channels=16,out_channels=3,stride=1,padding=1,kernel_size=(3,3))

    def forward(self,x):
        x=self.dconv1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.dp1(x)

        x=self.dconv2(x)
        x=self.bn2(x)
        x=self.relu2(x)
        x=self.dp2(x)

        x=self.dconv3(x)
        x=self.bn3(x)
        x=self.relu3(x)
        x=self.dp3(x)

        x=self.dconv4(x)
        x=self.bn4(x)
        x=self.relu4(x)
        x=self.dp4(x)

        x=self.dconv5(x)

        return x

# model=Reconsructor()
# summary(model,input_size=(256,64,64),batch_size=8,device='cpu')