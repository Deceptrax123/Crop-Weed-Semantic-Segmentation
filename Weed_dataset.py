import torch 
import torchvision 
import torchvision.transforms as T
from PIL import Image 
import numpy as np


class WeedDataset(torch.utils.data.Dataset):
    def _init__(self,paths):
        self.paths=paths

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,index):
        
        sample=self.paths[index]

        red,nir,ndvi,ground=sample[0],sample[1],sample[2],sample[3]
        red_img,nir_img,ndvi_img,ground_img=Image.open(red),Image.open(nir),Image.open(ndvi),Image.open(ground)
        red_np,nir_np,ndvi_np,ground_np=np.array(red_img),np.array(nir_img),np.array(ndvi_img),np.array(ground_img)

        #Prepare X
        X=np.stack((red_np,nir_np,ndvi_np),axis=-1)

        #Prepare Y->Convert to 3 channel one hot labels
        if 'weed' in red:

            pad=((0,0),(0,5))
            Y=np.pad(Y,pad,'constant',constant_values=(0,0))


        channels=3
        Y=np.eye(channels,dtype='unit8')[ground_np]

        #Perform transforms
        composed_transform_x=T.Compose([T.ToTensor(),T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        X_tensor=composed_transform_x(X)

        composed_transform_y=T.Compose([T.ToTensor()])
        Y_tensor=composed_transform_y(Y)

        return X_tensor,Y_tensor