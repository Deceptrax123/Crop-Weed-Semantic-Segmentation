import torch 
from torch import nn 
import torch.nn.functional as f

#input : BXCXHXW
#target : BXCXHXW

def overall_dice_score(input,target):

    probs=f.softmax(input,dim=1)
    preds=torch.argmax(probs,dim=1)
    predictions=torch.zeros_like(probs).scatter_(1,preds.unsqueeze(1),1.)
    

    pred_bflat=predictions.view(predictions.size(0),predictions.size(1)*predictions.size(2)*predictions.size(3))
    target_bflat=target.view(target.size(0),target.size(1)*target.size(2)*target.size(3))

    intersection=(pred_bflat*target_bflat).sum(dim=1)
    union=pred_bflat.sum(dim=1)+target_bflat.sum(dim=1)

    smooth=1

    dice=(2*(intersection)/(union+smooth)).mean()

    return dice 

def channel_dice_score(input,target):

    probs=f.softmax(input,dim=1)
    preds=torch.argmax(probs,dim=1)
    predictions=torch.zeros_like(probs).scatter_(1,preds.unsqueeze(1),1.)

    channels=input.size(1)-1 #ignore  background

    overall=0
    for i in range(channels):
        pred_channel=predictions[:,i+1,:,:]
        target_channel=target[:,i+1,:,:]

        pred_channel_bflat=pred_channel.view(pred_channel.size(0),pred_channel.size(1)*pred_channel.size(2))
        target_channel_bflat=target_channel.view(target_channel.size(0),target_channel.size(1)*target_channel.size(2))

        intersection=(pred_channel_bflat*target_channel_bflat).sum(dim=1)
        union=pred_channel_bflat.sum(dim=1)+target_channel_bflat.sum(dim=1)

        smooth=1

        dice=(2*(intersection)/(union+smooth)).mean()
        overall+=dice

    return overall/channels