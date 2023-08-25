import torch 
import torchvision.transforms.v2 as T 
import numpy as np 
from PIL import Image 
from base_model import EncDec
from data_script import read_file
import matplotlib.pyplot as plt

if __name__=='__main__':

    model=EncDec().to(device=torch.device('mps'))

    model.load_state_dict(torch.load("./models/run_2/model100.pth"))

    #get a test sample
    samples_train,samples_test=read_file()

    trans=T.Resize((1024,1024))
    red_img,nir_img,ndvi_img,ground_img=Image.open(samples_train[0][0]),Image.open(samples_train[0][1]),Image.open(samples_train[0][3]),Image.open(samples_train[0][2])
    red_img,nir_img,ndvi_img,ground_img=trans(red_img),trans(nir_img),trans(ndvi_img),trans(ground_img)
    red_np,nir_np,ndvi_np,ground_np=np.array(red_img),np.array(nir_img),np.array(ndvi_img),np.array(ground_img) #numpy arrays

    #Prepare X
    X=np.stack((red_np,nir_np,ndvi_np),axis=-1)
    X=X.astype(np.float32)


    #Prepare Y
    channels=3
    Y=np.eye(channels,dtype='uint8')[ground_np]
    Y=Y.astype(np.float32)


    totensor=T.Compose([T.ToImageTensor()])

    x_tensor=totensor(X)
    y_tensor=totensor(Y)

    mean=torch.mean(x_tensor,[1,2])
    std=torch.mean(x_tensor,[1,2])

    normalize=T.Normalize(mean=mean,std=std)
    x_tensor=normalize(x_tensor)

    x_tensor=x_tensor.to(device=torch.device('mps'))

    predictions=model(x_tensor)
    predictions=predictions.detach().numpy()

    pred_np=predictions.transpose(0,3,2,1)

    fig=plt.figure()

    ax1=fig.add_subplot(1,2,1)
    ax1.imshow(Y[0])
