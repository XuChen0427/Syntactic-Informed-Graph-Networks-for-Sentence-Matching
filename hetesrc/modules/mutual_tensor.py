import torch
import torch.nn as nn
import torch.nn.functional as F

class MutualTensor(nn.Module):
    def __init__(self,inputdim,args):
        super().__init__()
        self.input_dim = inputdim
        self.dropout = args.dropout
        self.mutual_dim = args.hidden_size

        self.mlp1 = nn.Linear(inputdim*2,inputdim)
        self.mlp2 = nn.Linear(inputdim,self.mutual_dim)

    def forward(self,a,b):
        mul_vec = torch.cat([a,b],dim=-1)

        hidden1 = self.mlp1(mul_vec)
        hidden1 = F.relu(hidden1)
        hidden1 = F.dropout(hidden1, self.dropout, self.training)
        out = self.mlp2(hidden1)
        return F.relu(out)