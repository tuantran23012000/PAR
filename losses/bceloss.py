import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.registry import LOSSES
from tools.function import ratio2weight


@LOSSES.register("bceloss")
class BCELoss(nn.Module):

    def __init__(self, sample_weight=None, size_sum=True, scale=None, tb_writer=None,samples_per_cls=None, no_of_classes=None,beta=None,gamma=None):
        super(BCELoss, self).__init__()

        self.sample_weight = sample_weight
        self.size_sum = size_sum
        self.hyper = 0.8
        self.smoothing = None
        self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.beta = beta
        self.gamma = gamma
        self.epsilon = 0.001
        self.alpha = 1000
    def ib_loss(self,input_values, ib):
        """Computes the focal loss"""
        loss = input_values * ib
        return loss.mean()
    def forward(self, logits, targets,features):
        # epsi = 1e-9
        logits = logits[0] # (BS x n_classes)
        # logit_p = torch.sum(torch.mul(targets,logits),0)/(torch.sum(targets,0)+epsi)
        # devi_p = torch.sqrt(torch.sum(torch.mul((logits-logit_p)**2+epsi,targets),0)/(torch.sum(targets,0)+epsi))
        # #print(devi_p)
        # logit_n = torch.sum(torch.mul(1-targets,logits),0)/(torch.sum(1-targets,0)+epsi)
        # devi_n = torch.sqrt(torch.sum(torch.mul((logits-logit_n)**2+epsi,1-targets),0)/(torch.sum(1-targets,0)+epsi))
        
        # #print(torch.mul(logits,devi_p)+logit_p)
        # #logits = torch.where(targets > 0.5, torch.mul(logits,devi_p)+logit_p,  torch.mul(logits,devi_n)+logit_n)
        # logits = torch.mul(torch.mul(logits,devi_p+epsi)+logit_p,targets) + torch.mul(torch.mul(logits,devi_n+epsi)+logit_n,1-targets)
        if self.smoothing is not None:
            targets = (1 - self.smoothing) * targets + self.smoothing * (1 - targets)

        loss_m = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        #print(logits.shape)
        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        if self.sample_weight is not None:
            sample_weight = ratio2weight(targets_mask, self.sample_weight)

            loss_m = (loss_m * sample_weight.cuda())
            loss = loss_m.sum(1).mean() if self.size_sum else loss_m.sum()
        else:

            effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
            weights = (1.0 - self.beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * self.no_of_classes
            sample_weight = torch.tensor(weights, device=logits.device).float()
            #sample_weight = ratio2weight(targets_mask, self.sample_weight)
            
            loss_m = F.binary_cross_entropy_with_logits(logits, targets,weight = sample_weight , reduction='none')
            grads = torch.sum(torch.abs(F.softmax(logits, dim=1) - targets_mask.cuda()),1) # N * 1
            ib = grads*features.reshape(-1)
            ib = self.alpha / (ib + self.epsilon)
            ib = ib.reshape(grads.size(0),1)
            loss = self.ib_loss(loss_m,ib)

        return [loss], [loss_m]