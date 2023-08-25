import torch 
import torchvision 
import torchvision.transforms.v2 as T
from PIL import Image 
import numpy as np


class WeedDataset(torch.utils.data.Dataset):
    def __init__(self,paths,training):
        self.paths=paths
        self.training=training

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,index):
        
        sample=self.paths[index]

        ndvi,nir,ground,red=sample[0],sample[1],sample[2],sample[3]
        red_img,nir_img,ndvi_img,ground_img=Image.open(red),Image.open(nir),Image.open(ndvi),Image.open(ground) #PIL objects

        trans=T.Resize((1024,1024))
        red_img,nir_img,ndvi_img,ground_img=trans(red_img),trans(nir_img),trans(ndvi_img),trans(ground_img)
        red_np,nir_np,ndvi_np,ground_np=np.array(red_img),np.array(nir_img),np.array(ndvi_img),np.array(ground_img) #numpy arrays

        #Prepare X
        X=np.stack((red_np,nir_np,ndvi_np),axis=-1)
        X=X.astype(np.float32)

        #Prepare Y
        channels=3
        Y=np.eye(channels,dtype='uint8')[ground_np]
        Y=Y.astype(np.float32)

        #Perform transforms and augmentation
        if(self.training):
            augmentation=T.Compose([T.RandomRotation(degrees=[45,60,15,30,90]),T.RandomHorizontalFlip(p=0.2),T.RandomVerticalFlip(p=0.2),T.ToImageTensor()])
        else:
            augmentation=T.Compose([T.ToImageTensor()])

        X_tensor=augmentation(X)

        mean_img=torch.mean(X_tensor,[1,2])
        std_img=torch.mean(X_tensor,[1,2])
        
        normalize=T.Normalize(mean=mean_img,std=std_img)

        X_tensor_norm=normalize(X_tensor)

        Y_tensor=augmentation(Y)

        return X_tensor_norm,Y_tensor
