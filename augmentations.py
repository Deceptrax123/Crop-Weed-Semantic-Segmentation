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

    ctr=180

    resize=T.CenterCrop((1024,1024))
    for i in test:
        ndvi,nir,red,ground_truth=i[0],i[1],i[3],i[2]
        ndvi,nir,red,ground_truth=Image.open(ndvi),Image.open(nir),Image.open(red),Image.open(ground_truth)

        ndvi,nir,red,ground_truth=resize(ndvi),resize(nir),resize(red),resize(ground_truth)
        ndvi_np,nir_np,red_np,ground_np=np.array(ndvi),np.array(nir),np.array(red),np.array(ground_truth)



        img=np.stack((ndvi_np,nir_np,red_np,ground_np),axis=-1)

        img=Image.fromarray(img)
        auguemtation=T.Compose([T.RandomRotation(degrees=(270,271))])

        img=auguemtation(img)

        img=np.array(img)

        ndvi_aug=Image.fromarray(img[:,:,0])
        nir_aug=Image.fromarray(img[:,:,1])
        red_aug=Image.fromarray(img[:,:,2])
        ground_aug=Image.fromarray(img[:,:,3])

        ndvi_aug.save("./data/Sequoia/SequoiaMulti_30/augmentations/samples/ndvi_aug_{no}.png".format(no=ctr))
        nir_aug.save("./data/Sequoia/SequoiaMulti_30/augmentations/samples/nir_aug_{no}.png".format(no=ctr))
        red_aug.save("./data/Sequoia/SequoiaMulti_30/augmentations/samples/red_aug_{no}.png".format(no=ctr))
        ground_aug.save("./data/Sequoia/SequoiaMulti_30/augmentations/annots/ground_aug_{no}.png".format(no=ctr))

        image_paths=[
            "./data/Sequoia/SequoiaMulti_30/augmentations/samples/ndvi_aug_{no}.png".format(no=ctr),"./data/Sequoia/SequoiaMulti_30/augmentations/samples/nir_aug_{no}.png ./data/Sequoia/SequoiaMulti_30/augmentations/annots/ground_aug_{no1}.png".format(no=ctr,no1=ctr),"./data/Sequoia/SequoiaMulti_30/augmentations/samples/red_aug_{no}.png".format(no=ctr)
          ]

        for k,j in enumerate(file_paths):
            f=open(j,'a')
            img_path=image_paths[k]

            f.write(img_path+"\n")

            f.close()
        ctr+=1
augment()