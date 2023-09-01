import torch 
import torch.nn.functional as F 
from torch.nn import Module 
import torch.nn as nn
from torch.autograd import Variable

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

            pred_batch=pred_channel.view(pred_channel.size(0),pred_channel.size(1)*pred_channel.size(2))
            target_batch=target_channel.view(target_channel.size(0),target_channel.size(1)*target_channel.size(2))

            intersection=(pred_batch*target_batch).sum(dim=1)
            union=pred_batch.sum(dim=1)+target_batch.sum(dim=1)
            
            smooth=1e-6

            dice=(2*(intersection))/(union+smooth).mean()
            l=1-dice

            loss+=l
        likelihood=torch.log1p(torch.cosh(loss))
        return likelihood.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha).to(device=torch.device('mps'))
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()