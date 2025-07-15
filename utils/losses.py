import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import utils.config as config


def convert_sigmoid_logits_to_binary_logprobs(logits):
    """Computes log(sigmoid(logits)), log(1-sigmoid(logits))."""
    log_prob = -F.softplus(-logits)
    log_one_minus_prob = -logits + log_prob
    return log_prob, log_one_minus_prob


def cross_entropy_loss(logits, labels, **kwargs):
    """ Modified cross entropy loss. """
 
    nll = F.log_softmax(logits, dim=-1)
    loss = -nll * labels
    loss = loss
    return loss.sum(dim=-1).mean()


def cross_entropy_loss_arc(logits, labels, **kwargs):
    """ Modified cross entropy loss. """
    f = kwargs['per']
    nll = F.log_softmax(logits, dim=-1)
    loss = -nll * labels * f #* kwargs['scale']
    # if kwargs['scale_flag']:
        # loss = loss * kwargs['scale']

    return loss.sum(dim=-1).mean()

class Plain(nn.Module):
    def forward(self, logits, labels, **kwargs):
        if config.loss_type == 'ce':
            loss = cross_entropy_loss(logits, labels, **kwargs)
        elif config.loss_type == 'ce_margin':
            loss = cross_entropy_loss_arc(logits, labels, **kwargs)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss *= labels.size(1)
        return loss

class GeneralizedCELoss(nn.Module):

    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
             
    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach()**self.q)*self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')

        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight

        return loss.mean()