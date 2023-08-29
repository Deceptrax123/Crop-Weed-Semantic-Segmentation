import torch 
from torch.nn import Conv2d,ConvTranspose2d,ReLU,BatchNorm2d,MaxPool2d,MaxUnpool2d,AdaptiveAvgPool2d,Upsample,Dropout2d,LeakyReLU
from torch.nn import Module 
from torchsummary import summary
from torch import nn

class Unet_encoding_block(Module):
    def __init__(self,features):
        super(Unet_encoding_block,self).__init__()

        self.conv1=Conv2d(in_channels=features,out_channels=2*features,stride=1,kernel_size=(3,3),padding=1)
        self.bn1=BatchNorm2d(2*features)
        self.relu1=ReLU()

        self.conv2=Conv2d(in_channels=2*features,out_channels=2*features,stride=1,kernel_size=(3,3),padding=1)
        self.bn2=BatchNorm2d(2*features)
        self.relu2=ReLU()

        self.apply(self._init_weights)
    
    def _init_weights(self,module):
        if isinstance(module,(nn.Conv2d,nn.BatchNorm2d)):
            if module.bias.data is not None :
                module.bias.data.zero_()
            else:
                nn.init.kaiming_normal_(module.weight.data,mode='fan_in',nonlinearity='relu')

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

        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,(nn.ConvTranspose2d,nn.BatchNorm2d)):
            if module.bias.data is not None :
                module.bias.data.zero_()
            else:
                nn.init.kaiming_normal_(module.weight.data,mode='fan_in',nonlinearity='relu')

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

        self.bconv1=Conv2d(in_channels=3,out_channels=8,stride=2,kernel_size=(3,3),padding=1)
        self.bn1=BatchNorm2d(8)
        self.relu1=LeakyReLU(negative_slope=0.2)
        self.dp1=Dropout2d()

        self.bconv2=Conv2d(in_channels=8,out_channels=16,stride=2,kernel_size=(3,3),padding=1)
        self.bn2=BatchNorm2d(16)
        self.relu2=LeakyReLU(negative_slope=0.2)
        self.dp2=Dropout2d()

        self.bconv3=Conv2d(in_channels=16,out_channels=32,stride=2,kernel_size=(3,3),padding=1)
        self.bn3=BatchNorm2d(32)
        self.relu3=LeakyReLU(negative_slope=0.2)
        self.dp3=Dropout2d()

        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,(nn.Conv2d,nn.BatchNorm2d)):
            if module.bias.data is not None :
                module.bias.data.zero_()
            else:
                nn.init.kaiming_normal_(module.weight.data,mode='fan_in',nonlinearity='leaky_relu')


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

        return x

class Reconsructor(Module):
    def __init__(self):
        super(Reconsructor,self).__init__()

        self.dconv1=ConvTranspose2d(in_channels=32,out_channels=16,stride=2,padding=1,output_padding=1,kernel_size=(3,3))
        self.bn1=BatchNorm2d(16)
        self.relu1=ReLU()
        self.dp1=Dropout2d()

        self.dconv2=ConvTranspose2d(in_channels=16,out_channels=8,padding=1,output_padding=1,kernel_size=(3,3),stride=2)
        self.bn2=BatchNorm2d(8)
        self.relu2=ReLU()
        self.dp2=Dropout2d()

        self.dconv3=ConvTranspose2d(in_channels=8,out_channels=3,kernel_size=(3,3),padding=1,output_padding=1,stride=2)

        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,(nn.ConvTranspose2d,nn.BatchNorm2d)):
            if module.bias.data is not None :
                module.bias.data.zero_()
            else:
                nn.init.kaiming_normal_(module.weight.data,mode='fan_in',nonlinearity='relu')

        
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

        return x

#model=Self_embedding_block()
#summary(model,input_size=(3,1024,1024),batch_size=8,device='cpu')