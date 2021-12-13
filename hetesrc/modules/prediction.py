

import torch
import torch.nn as nn
from functools import partial
from hetesrc.utils.registry import register
from . import Linear

registry = {}
register = partial(register, registry=registry)

class Prediction_Bert(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Dropout(args.dropout),
            Linear(args.hidden_size * 4, args.hidden_size, activations=True),
            nn.Dropout(args.dropout),
            Linear(args.hidden_size, args.num_classes),
        )

    def forward(self, x):
        return self.dense(x)

class Prediction_Bert_GAT(nn.Module):
    def __init__(self, args):
        super().__init__()
        #print("============")
        #print(args.num_classes)
        self.dense = nn.Sequential(
            nn.Dropout(args.dropout),
            Linear(args.hidden_size * 4, args.hidden_size, activations=True),
            nn.Dropout(args.dropout),
            Linear(args.hidden_size, args.num_classes),
        )


    def forward(self, a,b,res):
        #fusion = torch.cat([a, b, (a - b).abs(), a * b], dim=-1) + res
        fusion = torch.cat([a, b, a - b, a * b], dim=-1) + res

        return self.dense(fusion)



@register('simple')
class Prediction(nn.Module):
    def __init__(self, args, inp_features=2):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Dropout(args.dropout),
            Linear(args.hidden_size * inp_features *2, args.hidden_size, activations=True),
            nn.Dropout(args.dropout),
            Linear(args.hidden_size, args.num_classes),
        )

    def forward(self, a, b ,res):
        return self.dense(torch.cat([a, b ,res], dim=-1))


@register('full')
class AdvancedPrediction(Prediction):
    def __init__(self, args):
        super().__init__(args, inp_features=4)

    def forward(self, a, b,res):
        return self.dense(torch.cat([a, b, a - b, a * b , res], dim=-1))


@register('symmetric')
class SymmetricPrediction(AdvancedPrediction):
    def forward(self, a, b ,res):
        return self.dense(torch.cat([a, b, (a - b).abs(), a * b , res], dim=-1))
