import torch, math, numpy as np, scipy.sparse as sp
import torch.nn as nn, torch.nn.functional as F, torch.nn.init as init

import torch_geometric
from torch_geometric.nn import GCNConv
 
import torch_scatter, torch_sparse

class LEGCN(nn.Module):

    @staticmethod
    def line_expansion(data):
        V, E = data.edge_index
        E = E + V.max() # [V | E]
        num_ne_pairs = data.edge_index.shape[1]
        V_plus_E = E.max() + 1
        L1 = torch.stack([torch.arange(V.shape[0], device=V.device), V], -1)
        L2 = torch.stack([torch.arange(E.shape[0], device=E.device), E], -1)
        L = torch.cat([L1, L2], -1) # [2, |V| + |E|]
        L_T = torch.stack([L[1], L[0]], 0) # [2, |V| + |E|]
        ones = torch.ones(L.shape[1], device=L.device)
        adj, value = torch_sparse.spspmm(L, ones, L_T, ones, num_ne_pairs, V_plus_E, num_ne_pairs, coalesced=True)
        adj, value = torch_sparse.coalesce(adj, value, num_ne_pairs, num_ne_pairs, op="add")
        data.le_adj = adj
        return data

    def __init__(self, num_features, num_classes, args):
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of classes
        """
        super(LEGCN, self).__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(GCNConv(in_channels=num_features, out_channels=args.MLP_hidden))
        for l in range(args.All_num_layers - 2):
            self.layers.append(GCNConv(in_channels=args.MLP_hidden, out_channels=args.MLP_hidden))
        self.layers.append(GCNConv(in_channels=args.MLP_hidden, out_channels=num_classes))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data):
        """
        an l-layer GCN
        """
        le_adj = data.le_adj
        edge_index = data.edge_index
        x = data.x[edge_index[0]]
        for conv in self.layers[:-1]:
            x = F.relu(conv(x, le_adj))
        x = self.layers[-1](x, le_adj)
        x = torch_scatter.scatter(x, edge_index[0], dim=0, reduce='mean')
        return x
