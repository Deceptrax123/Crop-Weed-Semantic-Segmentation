import torch 
import torchvision.transforms.v2 as T 
from Weed_dataset import WeedDataset
from torch.utils.data import DataLoader
import numpy as np 
from PIL import Image 
from Base_paper.my_arch import EncDec
from data_script import read_file
import matplotlib.pyplot as plt

if __name__=='__main__':

    model=EncDec().to(device=torch.device('mps'))

    model.load_state_dict(torch.load("./models/run_2/model300.pth"))

    samples_train,samples_test=read_file()