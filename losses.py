import torch 
import torch.nn.functional as F 
from torch.nn import Module 

class DiceLoss(Module):
    def __init__(self):
        super(DiceLoss,self).__init__()
    
    def forward(self,inputs,targets):
        pred=F.softmax(inputs,dim=1)

        loss=0
        channels=inputs.size(1)

        for i in range(1,channels):
            pred_channel=pred[:,i,:,:]
            target_channel=targets[:,i,:,:]

            pred_batch=pred_channel.view(pred_channel.size(0),pred_channel.size(1)*pred_channel.size(2)*pred_channel.size(3))
            target_batch=target_channel.view(target_channel.size(0),target_channel.size(1)*target_channel.size(2)*target_channel.size(3))

            intersection=(pred_batch*target_batch).sum(dim=1)
            union=pred_batch.sum(dim=1)+target_batch.sum(dim=1)
            
            smooth=1

            dice=(2*(intersection))/(union+smooth).mean()
            l=1-dice

            loss+=l
        likelihood=torch.log1p(torch.cosh(loss))
        return likelihood.mean()