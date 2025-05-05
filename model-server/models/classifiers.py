import torch
from torch import nn
from models.dnf import DNF

class DNFClassifier(nn.Module):
    """基于DNF(析取范式)的分类器模型"""
    def __init__(self, num_preds, num_conjuncts, n_out, delta=0.01, weight_init_type="normal"):
        super(DNFClassifier, self).__init__()
        self.dnf = DNF(num_preds, num_conjuncts, n_out, delta, weight_init_type=weight_init_type)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.dnf(x))