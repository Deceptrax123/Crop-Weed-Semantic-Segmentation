import torch 
from torch.nn import Module,Conv2d,Softmax2d,ReLU,MaxPool2d,Dropout2d,Upsample,AdaptiveAvgPool2d,ConvTranspose2d,BatchNorm2d
from torchsummary import summary


class EncDec(Module):
    def __init__(self):
        super(EncDec,self).__init__()

        self.conv1=Conv2d(in_channels=3,out_channels=8,stride=2,kernel_size=(3,3),padding=1)
        self.bn1=BatchNorm2d(8)
        self.r1=ReLU()

        self.conv2=Conv2d(in_channels=8,out_channels=16,stride=2,kernel_size=(3,3),padding=1)
        self.bn2=BatchNorm2d(16)
        self.r2=ReLU()

        self.conv3=Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=2,padding=1)
        self.bn3=BatchNorm2d(32)
        self.r3=ReLU()

        self.conv4=Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=2,padding=1)
        self.bn4=BatchNorm2d(64)
        self.r4=ReLU()

        self.conv5=Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=2,padding=1)
        self.bn5=BatchNorm2d(128)
        self.r5=ReLU()

        self.conv6=Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),padding=1,stride=2)
        self.bn6=BatchNorm2d(256)
        self.r6=ReLU()

        self.conv7=Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),padding=1,stride=2)
        self.bn7=BatchNorm2d(512)
        self.r7=ReLU()

        self.dconv1=ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=(3,3),padding=1,stride=2,output_padding=1)
        self.bn8=BatchNorm2d(256)
        self.r8=ReLU()

        self.dconv2=ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=(3,3),padding=1,stride=2,output_padding=1)
        self.bn9=BatchNorm2d(128)
        self.r9=ReLU()

        self.dconv3=ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=(3,3),stride=2,padding=1,output_padding=1)
        self.bn10=BatchNorm2d(64)
        self.r10=ReLU()

        self.dconv4=ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=(3,3),stride=2,padding=1,output_padding=1)
        self.bn11=BatchNorm2d(32)
        self.r11=ReLU()

        self.dconv5=ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=(3,3),stride=2,padding=1,output_padding=1)
        self.bn12=BatchNorm2d(16)
        self.r12=ReLU()

        self.dconv6=ConvTranspose2d(in_channels=16,out_channels=8,padding=1,stride=2,kernel_size=(3,3),output_padding=1)
        self.bn13=BatchNorm2d(8)
        self.r13=ReLU()

        self.dconv7=ConvTranspose2d(in_channels=8,out_channels=3,padding=1,stride=2,kernel_size=(3,3),output_padding=1)

        self.classifier=Softmax2d()
    
    def forward(self,x):

        #encoding layers
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.r1(x)

        x=self.conv2(x)
        x=self.bn2(x)
        x=self.r2(x)

        x=self.conv3(x)
        x=self.bn3(x)
        x=self.r3(x)

        x=self.conv4(x)
        x=self.bn4(x)
        x=self.r4(x)

        x=self.conv5(x)
        x=self.bn5(x)
        x=self.r5(x)

        x=self.conv6(x)
        x=self.bn6(x)
        x=self.r6(x)

        x=self.conv7(x)
        x=self.bn7(x)
        x=self.r7(x)

        #decoding layers
        x=self.dconv1(x)
        x=self.bn8(x)
        x=self.r8(x)

        x=self.dconv2(x)
        x=self.bn9(x)
        x=self.r9(x)

        x=self.dconv3(x)
        x=self.bn10(x)
        x=self.r10(x)

        x=self.dconv4(x)
        x=self.bn11(x)
        x=self.r11(x)

        x=self.dconv5(x)
        x=self.bn12(x)
        x=self.r12(x)

        x=self.dconv6(x)
        x=self.bn13(x)
        x=self.r13(x)

        x=self.dconv7(x)

        return x
