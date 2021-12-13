import torch
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import init
import math

class AttentionLayer(nn.Module):
    def __init__(self,input_dim):
        super(AttentionLayer, self).__init__()
        self.Q_linear = nn.Linear(input_dim,input_dim)
        self.K_linear = nn.Linear(input_dim,input_dim)
        self.V_linear = nn.Linear(input_dim,input_dim)
        self.temperature = nn.Parameter(torch.tensor(1 / math.sqrt(input_dim)))

    def forward(self,Q,K,V):
        Q = self.Q_linear(Q)
        #[n,1,D]
        K = self.K_linear(K)
        #[n,5,D]
        V = self.V_linear(V)

        attn = torch.matmul(Q, K.transpose(1, 2)) * self.temperature
        #[n,1,5]
        #mask = torch.matmul(mask_a.float(), mask_b.transpose(1, 2).float()).byte().bool()

        #attn.masked_fill_(~mask, -1e7)

        attn = F.softmax(attn, dim=-1)

        feature = torch.matmul(attn, V).squeeze(1)
        return feature

class GCN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GCN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.GCNConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.GCNConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.GCNConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        #x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x

class SAGE(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(SAGE, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.SAGEConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.SAGEConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.SAGEConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.SAGEConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        #x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x


class HeteGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_type,
                 feature_pre=True, layer_num=2, dropout=True, dropout_rate=0.2,**kwargs):
        super(HeteGAT, self).__init__()
        self.feature_pre = feature_pre
        self.n_type = n_type
        self.layer_num = layer_num
        self.dropout = dropout
        self.middle_dim = hidden_dim*4
        self.LayerNorm = nn.LayerNorm(hidden_dim,eps=1e-12)
        self.dropout_rate = dropout_rate
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, hidden_dim)
            #self.conv_first = tg.nn.GATConv(hidden_dim, hidden_dim)

        self.conv_hidden = nn.ModuleList([nn.ModuleList([tg.nn.GATConv(hidden_dim, hidden_dim,dropout=dropout_rate) for j in range(n_type)]) for i in range(layer_num)])
        self.attention_layer = nn.ModuleList([AttentionLayer(hidden_dim) for i in range(layer_num)])
        self.conv_res = nn.ModuleList([Output(hidden_dim, self.middle_dim,dropout_rate=dropout_rate) for i in range(layer_num)])
        self.out = nn.Linear(hidden_dim,output_dim)



    def forward(self, x, edge_index):
        #x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
            x = F.leaky_relu(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout_rate,training=self.training)

        #pre_x = x
        for i in range(self.layer_num):
            pre_x = x
            x_type = []
            for j in range(self.n_type):
                x_type.append(self.conv_hidden[i][j](x, edge_index[j]).unsqueeze(1))
                if self.dropout:
                    x_type[j] = F.dropout(x_type[j], p=self.dropout_rate,training=self.training)

            #[N,N_type,D]
            x_t = torch.cat(x_type,dim=1)
            #print(x_t.shape)
            x_att = self.attention_layer[i](x.unsqueeze(1),x_t,x_t)
            # print(x_att.shape)
            # print(x.shape)
            # exit(0)

            x = F.gelu(x_att)
            x = self.LayerNorm(pre_x+x)


            x = self.conv_res[i](x)
            if self.dropout:
                x = F.dropout(x, p=self.dropout_rate,training=self.training)


        x = self.out(x)
        if self.dropout:
            F.dropout(x, p=self.dropout_rate,training=self.training)
        x = F.gelu(x)

        #x = F.normalize(x, p=2, dim=-1)
        return x

class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, dropout_rate=0.2,**kwargs):
        super(GAT, self).__init__()
        self.feature_pre = feature_pre
        self.middle_dim = hidden_dim*4
        self.layer_num = layer_num
        self.dropout = dropout
        self.LayerNorm = nn.LayerNorm(hidden_dim,eps=1e-12)
        self.dropout_rate = dropout_rate
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, hidden_dim)
            #self.conv_first = tg.nn.GATConv(hidden_dim, hidden_dim)

        self.conv_hidden = nn.ModuleList([tg.nn.GATConv(hidden_dim, hidden_dim,dropout=dropout_rate) for i in range(layer_num)])
        self.conv_res = nn.ModuleList([Output(hidden_dim, self.middle_dim,dropout_rate=dropout_rate) for i in range(layer_num)])
        #self.conv_out = tg.nn.GATConv(hidden_dim, hidden_dim)

        self.out = nn.Linear(hidden_dim,output_dim)



    def forward(self, x, edge_index):
        #x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
            x = F.leaky_relu(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout_rate,training=self.training)

        pre_x = x
        for i in range(self.layer_num):
            x = self.conv_hidden[i](x, edge_index)
            if self.dropout:
                x = F.dropout(x, p=self.dropout_rate,training=self.training)

            x = F.gelu(x)
            x = self.LayerNorm(pre_x+x)


            x = self.conv_res[i](x)
            if self.dropout:
                x = F.dropout(x, p=self.dropout_rate,training=self.training)

        x = self.out(x)
        if self.dropout:
            F.dropout(x, p=self.dropout_rate,training=self.training)
        x = F.relu(x)

        #x = F.normalize(x, p=2, dim=-1)
        return x

    def GetAtt(self,x,edge_index):
        if self.feature_pre:
            x = self.linear_pre(x)
            x = F.leaky_relu(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout_rate,training=self.training)

        pre_x = x
        for i in range(self.layer_num):
            x,att = self.conv_hidden[i](x, edge_index,return_attention_weights=True )

            if i == self.layer_num -1:
                return att[0],att[1]
            if self.dropout:
                x = F.dropout(x, p=self.dropout_rate,training=self.training)

            x = F.gelu(x)
            x = self.LayerNorm(pre_x+x)


            x = self.conv_res[i](x)
            if self.dropout:
                x = F.dropout(x, p=self.dropout_rate,training=self.training)



class Output(nn.Module):
    def __init__(self, hidden_size,middle_size,dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.dense_mid = nn.Linear(hidden_size, middle_size)

        self.LayerNorm = nn.LayerNorm(hidden_size,eps=1e-12)
        self.dense_out = nn.Linear(middle_size, hidden_size)

    def forward(self, hidden_states):
        mid_states = self.dense_mid(hidden_states)
        mid_states = F.dropout(mid_states,p=self.dropout_rate, training=self.training)
        mid_states  = F.gelu(mid_states)
        mid_states = self.dense_out(mid_states)
        hidden_states = self.LayerNorm(hidden_states + mid_states)
        return hidden_states