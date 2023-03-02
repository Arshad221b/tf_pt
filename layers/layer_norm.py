import torch.nn as nn 


class LayerNorm(nn.Module):
    def __init__(self) -> None:
        super(LayerNorm).__init__()
        self.norm = nn.LayerNorm()
        
        
    def forward(self):
        x = self.norm(x)
        
        return x
    
    
    