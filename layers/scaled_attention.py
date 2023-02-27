import torch.nn as nn 
import numpy as np 
import torch
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super(ScaledDotProductAttention).__init__()
        self.softmax = nn.Softmax(dim= -1)
    
    def forward(self, q, k, v, mask = None):
        '''
        Input has 4 dim
        [batch_size, heads, length, d_tensor]
        batch_size = self explanatory 
        heads = No of heads in the attention layer
        length = length of each input vector
        d_tensor = the embedding dimension
        '''
        batch_size, heads, length, d_tensor = k.size()
        
        # step 1: take dot product of Q and K
        k_t = torch.t(k)
        
        score = (q @ k_t)/math.sqrt(d_tensor)
        
        
        # if mask != None:
        #     score = score.masked_fill(mask == 0, -10000)
        
        score = self.softmax(score)
        
        v = score @ v 
        
        return v, score 
        
        
        
        
        
        
        
        
        
        