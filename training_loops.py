import torch 
import torchvision
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

def train_step():
    epoch_loss=0

    for step,(x_sample,y_sample) in enumerate(train_loader):
        x_sample=x_sample.to(device=device)
        y_sample=y_sample.to(device=device)

        #model training
        model.zero_grad()
        predictions=model(x_sample)
        loss=loss_function(predictions,y_sample)

        loss.backward()
        model_optimizer.step()

        epoch_loss+=loss.item()
    reduced_loss=epoch_loss/train_steps
    return  reduced_loss

def test_step():
    epoch_loss=0

    for step,(x_sample,y_sample) in enumerate(test_loader):
        x_sample=x_sample.to(device=device)
        y_sample=y_sample.to(device=device)

        #test set evaluations
        predictions=model(x_sample)
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
    num_epochs=50
    loss_function=nn.CrossEntropyLoss()

    #set model and optimizers
    model=EncDec().to(device=device)

    #weight initializer
    initialize_weights(model)

    model_optimizer=torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.999))

    train_steps=(len(train)+params['batch_size']-1)//params['batch_size']
    test_steps=(len(test)+params['batch_size']-1)//params['batch_size']

    training_loop()