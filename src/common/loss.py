import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def align_loss_function_torch(x, y, alpha=2.0, beta=1.0):
    # x, y: (b, Q, D)
    dist_ = ((x - y) ** 2).sum(dim=2)  # (b, Q)
    dist_ = dist_.pow(alpha / 2.0)     # (b, Q)
    dist_ = dist_.mean(dim=1)         # (b,)
    return dist_.mean(dim=0)          # (,)


def uni_loss_function_torch(x, t=2., beta=1.):
    # x: (b, Q, D)
    b, q, d = x.shape
    x = x.permute(1, 0, 2)  # (Q, b, D)
    sq_dist = torch.sum((x[:, :, None] - x[:, None]) ** 2, dim=3)  # (Q, b, b)
    sq_dist = sq_dist.view(q, b * b)  # (Q, b*b)
    sq_dist_flat = sq_dist.view(-1)   # (Q*b*b)
    loss = torch.logsumexp(-t * sq_dist_flat, dim=0) - torch.log(torch.tensor(b * b * q, dtype=torch.float))
    return loss

def uni_loss_function_torch_sr(x, t=2., beta=1.):
    # x: (b, f)
    b, qd = x.shape
    sq_dist = torch.sum((x[:, None] - x[None,...]) ** 2, dim=2)  # (Q, b, b)
    sq_dist = sq_dist.view(b * b, )  # (Q, b*b)

    loss = torch.logsumexp(-t * sq_dist, dim=0) - torch.log(torch.tensor(b * b, dtype=torch.float))
    return loss

def lalign(x, y, alpha=2):
    return (x - y).norm(dim=1).pow(alpha).mean()

def lalign_reduce(x, y, weight=None, alpha=2):
    distances = (x - y).norm(dim=1).pow(alpha)
    if weight is None:
        return distances.mean()
    elif weight.sum() == 0:
        return 0
    else:
        assert weight.dim() == 2 and weight.shape[1] == 1 or weight.dim() == 1
        weight = weight.view(-1)
        return (distances * weight).sum() / weight.sum()
    
    
def lalign_constrain_reduce(x, y, prior_distance=None, alpha=2):
    distances = (x - y).norm(dim=1, dtype=torch.float32)
    
    if prior_distance is not None:
        prior_distance = prior_distance.squeeze(-1)
        loss = ((distances - prior_distance).pow(alpha)).mean()
    else:
        loss = (x - y).norm(dim=1).pow(alpha).mean()
    
    return loss
    

def lunif(x, t=2):
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()

def split_latent_feat_torch(latent_feat):
    split_size = latent_feat.size(0) // 2
    latent_feat_x, latent_feat_y = torch.split(latent_feat, split_size, dim=0)
    return latent_feat_x, latent_feat_y


def reduce_loss(logits,mask,labels,weight_pow = 0.5):

    batch_size, length, dim = logits.shape
    masked_loss = F.cross_entropy(logits.view(-1, dim), labels.view(-1), reduction='none',label_smoothing = 0.05)
    masked_loss = masked_loss.view(batch_size, length)
    masked_loss = masked_loss * mask
    masked_loss = masked_loss.sum(dim=1) / mask.sum(dim=1)
    weights = mask.sum(dim=1).pow(weight_pow)
    weighted_loss = (masked_loss * weights).sum() / weights.sum()

    return weighted_loss