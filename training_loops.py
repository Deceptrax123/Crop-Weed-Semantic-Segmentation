import torch 
import torchvision.transforms as T
from torch.utils.data import DataLoader
from Weed_dataset import WeedDataset
from base_model import EncDec
from initializer import initialize_weights
from time import time 
from torch import nn
import torch.multiprocessing
import wandb
from data_script import read_file
import matplotlib.pyplot as plt
import numpy as np 

def compute_weights(y_sample):
    #size-(batch_size,3,1024,1024)

    #get counts of 1s 
    channels=3

    counts=list()
    for i in range(channels):
        spectral_region=y_sample[:,i,:,:]

        ones=(spectral_region==1.).sum()
        counts.append(ones)

    total_pixels=y_sample.size(0)*channels*1024*1024

    counts=np.array(counts)
    weights=counts/total_pixels

    inverse=1/weights
    inverse=inverse.astype(np.float32)
    return inverse

def train_step():
    epoch_loss=0

    for step,(x_sample,y_sample) in enumerate(train_loader):
        weights=compute_weights(y_sample)
        x_sample=x_sample.to(device=device)
        y_sample=y_sample.to(device=device)
        weights=torch.from_numpy(weights).to(device=device)

        #model training
        model.zero_grad()
        predictions=model(x_sample)

        #compute loss function and perform backpropagation
        loss_function=nn.CrossEntropyLoss(weight=weights)
        loss=loss_function(predictions,y_sample)

        loss.backward()
        model_optimizer.step()

        epoch_loss+=loss.item()
    reduced_loss=epoch_loss/train_steps
    return  reduced_loss

def test_step():
    epoch_loss=0

    for step,(x_sample,y_sample) in enumerate(test_loader):
        #compute sample weights
        weights=compute_weights(y_sample)

        x_sample=x_sample.to(device=device)
        y_sample=y_sample.to(device=device)
        weights=torch.from_numpy(weights).to(device=device)

        #test set evaluations
        predictions=model(x_sample)

        #compute loss
        loss_function=nn.CrossEntropyLoss(weight=weights)
        loss=loss_function(predictions,y_sample)

        epoch_loss+=loss.item()
    reduced_loss=epoch_loss/test_steps

    return reduced_loss

def training_loop():
    for epoch in range(num_epochs):

        model.train(True) #train mode
        train_loss=train_step()
        model.eval() #eval mode

        test_loss=test_step()

        print('Epoch {epoch}'.format(epoch=epoch+1))
        print('Train Loss : {tloss}'.format(tloss=train_loss))
        print("Test Loss : {teloss}".format(teloss=test_loss))

        wandb.log({
            "Train Loss":train_loss,
            "Test Loss":test_loss
        })

        #checkpoints
        if((epoch+1)%10==0):
            path="./models/run_2/model{epoch}.pth".format(epoch=epoch+1)
            torch.save(model.state_dict(),path)


if __name__=='__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')

    train,test=read_file()

    params={
        'batch_size':8,
        'shuffle':True,
        'num_workers':0
    }

    train_set=WeedDataset(paths=train,training=True)
    test_set=WeedDataset(paths=test,training=False)

    wandb.init(
        project='weed-detection',
        config={
            "architecture":'DL based models',
            "dataset":"weedNet datatset",
        },
    )

    #train and test loaders
    train_loader=DataLoader(train_set,**params)
    test_loader=DataLoader(test_set,**params)

    #device
    if torch.backends.mps.is_available():
        device=torch.device("mps")
    else:
        device=torch.device("cpu")


    #Hyperparameters
    lr=0.001
    num_epochs=100
    loss_function=nn.CrossEntropyLoss()

    #set model and optimizers
    model=EncDec().to(device=device)

    #weight initializer
    initialize_weights(model)

    model_optimizer=torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.999))

    train_steps=(len(train)+params['batch_size']-1)//params['batch_size']
    test_steps=(len(test)+params['batch_size']-1)//params['batch_size']

    training_loop()   