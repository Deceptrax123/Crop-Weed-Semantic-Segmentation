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
        red_img,nir_img,ndvi_img=trans(red_img),trans(nir_img),trans(ndvi_img),trans(ground_img)
        red_np,nir_np,ndvi_np,ground_np=np.array(red_img),np.array(nir_img),np.array(ndvi_img),np.array(ground_img) #numpy arrays

        #Prepare X
        X=np.stack((red_np,nir_np,ndvi_np),axis=-1)

        #Prepare Y
        channels=3
        Y=np.eye(channels,dtype='unit8')[ground_np]

        #Perform transforms and augmentation
        if(self.training):
            augmentation=T.Compose([T.RandomRotation(degrees=45),T.RandomHorizontalFlip(p=0.5),T.RandomVerticalFlip(p=0.5),T.ToTensor()])
        else:
            augmentation=T.Compose([T.ToTensor()])
        
        normalize=T.Normalize((0,0,0),(1,1,1))

        X_tensor=augmentation(X)
        X_tensor_norm=normalize(X_tensor)

        Y_tensor=augmentation(Y)

        return X_tensor_norm,Y_tensor
