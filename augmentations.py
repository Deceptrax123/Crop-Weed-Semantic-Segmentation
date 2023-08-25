import torch 
import numpy as np 
import torchvision 
import torchvision.transforms.v2 as T 
from torchvision.utils import save_image
from data_script import read_file
from PIL import Image

def augment():
    train,test=read_file()

    file_paths=[
        "./data/Sequoia/SequoiaMulti_30/trainNDVI.txt","./data/Sequoia/SequoiaMulti_30/trainNir.txt","./data/Sequoia/SequoiaMulti_30/trainRed.txt"
    ]
    
    for ctr,i in enumerate(train):
        ndvi,nir,red,ground_truth=i[0],i[1],i[3],i[2]
        ndvi,nir,red,ground_truth=Image.open(ndvi),Image.open(nir),Image.open(red),Image.open(ground_truth)
        auguemtation=T.Compose([T.RandomHorizontalFlip(p=0.2),T.RandomVerticalFlip(p=0.2),T.RandomRotation(degrees=[45,60])])

        ndvi_aug,nir_aug,red_aug,ground_aug=auguemtation(ndvi),auguemtation(nir),auguemtation(red),auguemtation(ground_truth)

        ndvi_aug.save("./data/Sequoia/SequoiaMulti_30/augmentations/samples/ndvi_aug_{no}.png".format(no=ctr))
        nir_aug.save("./data/Sequoia/SequoiaMulti_30/augmentations/samples/nir_aug_{no}.png".format(no=ctr))
        red_aug.save("./data/Sequoia/SequoiaMulti_30/augmentations/samples/red_aug_{no}.png".format(no=ctr))
        ground_aug.save("./data/Sequoia/SequoiaMulti_30/augmentations/annots/ground_aug_{no}.png".format(no=ctr))

        image_paths=[
            "./data/Sequoia/SequoiaMulti_30/augmentations/samples/ndvi_aug_{no}.png".format(no=ctr),"./data/Sequoia/SequoiaMulti_30/augmentations/samples/nir_aug_{no}.png ./data/Sequoia/SequoiaMulti_30/augmentations/annots/ground_aug_{no1}.png".format(no=ctr,no1=ctr),"./data/Sequoia/SequoiaMulti_30/augmentations/samples/red_aug_{no}.png".format(no=ctr),
        ]

        for k,j in enumerate(file_paths):
            f=open(j,'a')
            img_path=image_paths[k]

            f.write(img_path+"\n")

            f.close()

#augment()