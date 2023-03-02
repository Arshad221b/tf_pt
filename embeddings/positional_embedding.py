import torch.nn as nn
import numpy as np 
import math


class PositionalEmbedding(nn.Module):
    '''
    following the paper, the encoding can be described as, 
    
    for even i, 
    PE(pos i) = sin(pos/ 10000^(2i/d_model))
    for odd i, 
    PE(pos i) = cos(pos/ 10000^(2i/d_model))
     
    '''
    def __init__(self, pos, dim, d_model) -> None:
        super(PositionalEmbedding, self).__init__()
        self.pos = pos 
        self.dim = dim
        self.d_model = d_model 
    
    def forward(self):
        pe = []*self.dim
        for i in range(0, self.dim, 2):
            pe[0][i] = math.sin(self.pos/(10000 ** ((2* i)/self.d_model)))
            pe[0][i+1] = math.cos(self.pos/(10000 ** ((2* i)/self.d_model)))
            
        return pe