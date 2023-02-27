import torch.nn as nn 

class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden) -> None:
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(d_model, d_hidden)
        self.l2 = nn.Linear(d_hidden, d_model)
        self.relu = nn.ReLU()
    
    def forward(self):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        
        return x 
    
    