# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
if device != "cpu":
    torch.cuda.set_device(0)

class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(support_embs, query_embs, support_y, query_y, n_support):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    
    support_embs_cpu = support_embs.to('cpu')
    query_embs_cpu = query_embs.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.tensor(list(query_y))
    n_classes = len(classes)
    n_query = 1

    support_idxs = [0 for i in range(len(query_y))]
    for idx in range(len(query_y)):
        cl = query_y[idx]
        support_idx = []
        for i in range(len(support_y)):
            if cl == support_y[i]:
                support_idx.append(i)
        support_idxs[idx] = torch.tensor(support_idx)
    support_idxs = torch.stack(support_idxs)

    prototypes = torch.stack([support_embs_cpu[idx_list].mean(0) for idx_list in support_idxs]).to(device)
    
    dists = euclidean_dist(query_embs, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)


    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long().to(device)

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    y_hat = y_hat.squeeze()
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val,  acc_val