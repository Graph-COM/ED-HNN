import torch

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from models.mlp import MLP

import numpy as np
import math

import torch_scatter

class UniGCNIIConv(nn.Module):
    def __init__(self, args, in_features, out_features):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.args = args

    def reset_parameters(self):
        self.W.reset_parameters()
        
    def forward(self, X, vertex, edges, alpha, beta, X0):
        N = X.shape[-2]

        Xve = X[..., vertex, :] # [nnz, C]
        Xe = torch_scatter.scatter(Xve, edges, dim=-2, reduce='mean') # [E, C], reduce is 'mean' here as default

        Xev = Xe[..., edges, :] # [nnz, C]
        Xv = torch_scatter.scatter(Xev, vertex, dim=-2, reduce='mean', dim_size=N) # [N, C]

        X = Xv 

        if self.args.UniGNN_use_norm:
            X = normalize_l2(X)

        Xi = (1-alpha) * X + alpha * X0
        X = (1-beta) * Xi + beta * self.W(Xi)


        return X

    def flops(self, X, vertex, edges):
        flops = vertex.shape[0] + edges.shape[0] + edges.shape[0] + vertex.shape[0] # scatter
        flops += np.prod(X.shape) # init connection
        flops += np.prod(X.shape[:-1]) * self.in_features * self.out_features # linear
        return flops


class UniGCNII(nn.Module):
    def __init__(self, num_features, num_classes, args):
        """UniGNNII

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        nlayer = args.All_num_layers
        nhid = args.MLP_hidden
        nhead = args.All_num_layers

        self.nhid = nhid
        self.in_features = num_features
        self.out_features = num_classes

        nhid = args.MLP_hidden
        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act['relu'] # Default relu
        self.input_drop = nn.Dropout(0.6) # 0.6 is chosen as default
        self.dropout = nn.Dropout(0.2) # 0.2 is chosen for GCNII
        self.alpha = args.restart_alpha

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(num_features, nhid))
        for _ in range(args.All_num_layers):
            self.convs.append(UniGCNIIConv(args, nhid, nhid))
        self.convs.append(torch.nn.Linear(nhid, num_classes))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())
        self.dropout = nn.Dropout(0.2) # 0.2 is chosen for GCNII

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
    def forward(self, data):
        x = data.x
        V, E = data.edge_index[0], data.edge_index[1]
        lamda, alpha = 0.5, self.alpha
        x = self.dropout(x)
        x = F.relu(self.convs[0](x))
        x0 = x 
        for i,con in enumerate(self.convs[1:-1]):
            x = self.dropout(x)
            beta = math.log(lamda/(i+1)+1)
            x = F.relu(con(x, V, E, alpha, beta, x0))
        x = self.dropout(x)
        x = self.convs[-1](x)
        return x

    def flops(self, data):
        x = data.x
        V, E = data.edge_index[0], data.edge_index[1]
        flops = self.in_features * self.nhid * np.prod(x.shape[:-1]) # linear
        flops += self.nhid * np.prod(x.shape[:-1]) # non-linear
        for i,con in enumerate(self.convs[1:-1]):
            flops += con.flops(x, V, E) # conv
            flops += self.nhid * np.prod(x.shape[:-1]) # non-linear
        flops = self.out_features * self.nhid * np.prod(x.shape[:-1]) # linear
        return flops
