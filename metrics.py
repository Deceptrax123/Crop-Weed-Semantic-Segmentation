import torch 
from torch import nn 
import torch.nn.functional as f

def dice_score(input,target):

    predictions=f.softmax(input)

    #intersection
    intersection=predictions*target
    intersection=torch.sum(intersection,dim=3)
    intersection=torch.sum(intersection,dim=3)

    #Union
    union1=predictions*predictions
    union1=torch.sum(union1,dim=3)
    union1=torch.sum(union1,dim=2)

    union2=target*target
    union2=torch.sum(union2,dim=3)
    union2=torch.sum(union2,dim=2)

    smooth=1

    dice=2.*(intersection)/(union1+union2+smooth)

    effective_dice=dice[:,1:]

    dice_score=torch.sum(effective_dice)/effective_dice.size(0)

    return dice_score