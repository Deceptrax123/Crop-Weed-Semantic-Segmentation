import torch 
from torch.nn import Module,Conv2d,Softmax2d,ReLU,MaxPool2d,Dropout2d,Upsample,AdaptiveAvgPool2d,ConvTranspose2d,BatchNorm2d
from torchsummary import summary


class MyArch_Dilated(Module):
    def __init__(self):
        super(MyArch_Dilated,self).__init__()

        self.conv1=Conv2d(in_channels=3,out_channels=8,stride=2,kernel_size=(3,3),padding=2,dilation=2)
        self.bn1=BatchNorm2d(8)
        self.r1=ReLU()

        self.conv2=Conv2d(in_channels=8,out_channels=16,stride=2,kernel_size=(3,3),padding=2,dilation=2)
        self.bn2=BatchNorm2d(16)
        self.r2=ReLU()

        self.conv3=Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=2,padding=2,dilation=2)
        self.bn3=BatchNorm2d(32)
        self.r3=ReLU()

        self.conv4=Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=2,padding=2,dilation=2)
        self.bn4=BatchNorm2d(64)
        self.r4=ReLU()

        self.conv5=Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=2,padding=2,dilation=2)
        self.bn5=BatchNorm2d(128)
        self.r5=ReLU()

        self.conv6=Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),padding=2,stride=2,dilation=2)
        self.bn6=BatchNorm2d(256)
        self.r6=ReLU()

        self.conv7=Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),padding=2,stride=2,dilation=2)
        self.bn7=BatchNorm2d(512)
        self.r7=ReLU()

        self.dconv1=ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=(3,3),padding=2,stride=2,output_padding=1,dilation=2)
        self.bn8=BatchNorm2d(256)
        self.r8=ReLU()

        self.dconv2=ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=(3,3),padding=2,stride=2,output_padding=1,dilation=2)
        self.bn9=BatchNorm2d(128)
        self.r9=ReLU()

        self.dconv3=ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=(3,3),stride=2,padding=2,output_padding=1,dilation=2)
        self.bn10=BatchNorm2d(64)
        self.r10=ReLU()

        self.dconv4=ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=(3,3),stride=2,padding=2,output_padding=1,dilation=2)
        self.bn11=BatchNorm2d(32)
        self.r11=ReLU()

        self.dconv5=ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=(3,3),stride=2,padding=2,output_padding=1,dilation=2)
        self.bn12=BatchNorm2d(16)
        self.r12=ReLU()

        self.dconv6=ConvTranspose2d(in_channels=16,out_channels=8,padding=2,stride=2,kernel_size=(3,3),output_padding=1,dilation=2)
        self.bn13=BatchNorm2d(8)
        self.r13=ReLU()

        self.dconv7=ConvTranspose2d(in_channels=8,out_channels=3,padding=2,stride=2,kernel_size=(3,3),output_padding=1,dilation=2)

        #dropoouts
        self.dp1=Dropout2d()
        self.dp2=Dropout2d()
        self.dp3=Dropout2d()
        self.dp4=Dropout2d()
        self.dp5=Dropout2d()
        self.dp6=Dropout2d()
        self.dp7=Dropout2d()
        self.dp8=Dropout2d()
        self.dp9=Dropout2d()
        self.dp10=Dropout2d()
        self.dp11=Dropout2d()
        self.dp12=Dropout2d()
        self.dp13=Dropout2d()

        self.classifier=Softmax2d()
    
    def forward(self,x):
        #encoding layers
        x1=self.conv1(x)
        x=self.bn1(x1)
        x=self.r1(x)
        x=self.dp1(x)

        x2=self.conv2(x)
        x=self.bn2(x2)
        x=self.r2(x)
        x=self.dp2(x)

        x3=self.conv3(x)
        x=self.bn3(x3)
        x=self.r3(x)
        x=self.dp3(x)

        x4=self.conv4(x)
        x=self.bn4(x4)
        x=self.r4(x)
        x=self.dp4(x)

        x5=self.conv5(x)
        x=self.bn5(x5)
        x=self.r5(x)
        x=self.dp5(x)

        x6=self.conv6(x)
        x=self.bn6(x6)
        x=self.r6(x)
        x=self.dp6(x)

        x7=self.conv7(x)
        x=self.bn7(x7)
        x=self.r7(x)
        x=self.dp7(x)

        x=self.dconv1(x)
        xcat1=torch.add(x,x6)
        x=self.bn8(xcat1)
        x=self.r8(x)
        x=self.dp8(x)

        x=self.dconv2(x)
        xcat2=torch.add(x,x5)
        x=self.bn9(xcat2)
        x=self.r9(x)
        x=self.dp9(x)

        x=self.dconv3(x)
        xcat3=torch.add(x,x4)
        x=self.bn10(xcat3)
        x=self.r10(x)
        x=self.dp10(x)

        x=self.dconv4(x)
        xcat4=torch.add(x,x3)
        x=self.bn11(xcat4)
        x=self.r11(x)
        x=self.dp11(x)

        x=self.dconv5(x)
        xcat5=torch.add(x,x2)
        x=self.bn12(xcat5)
        x=self.r12(x)
        x=self.dp12(x)

        x=self.dconv6(x)
        xcat6=torch.add(x,x1)
        x=self.bn13(xcat6)
        x=self.r13(x)
        x=self.dp13(x)

        x=self.dconv7(x)


        return x

#model=MyArch_Dilated()
#summary(model,input_size=(3,1024,1024),batch_size=8,device='cpu')