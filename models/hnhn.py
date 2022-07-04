import torch
import torch_scatter
from torch import Tensor
from torch.nn import Linear
from torch.nn import Parameter

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_add, scatter
from torch_geometric.typing import Adj, Size, OptTensor
from typing import Optional

import math
import numpy as np

# Method for initialization
def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class HNHNConv(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, nonlinear_inbetween=True,
                 concat=True, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HNHNConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.nonlinear_inbetween = nonlinear_inbetween

        # preserve variable heads for later use (attention)
        self.heads = heads
        self.concat = True
        # self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_v2e = Linear(in_channels, hidden_channels)
        self.weight_e2v = Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight_v2e.reset_parameters()
        self.weight_e2v.reset_parameters()
        # glorot(self.weight_v2e)
        # glorot(self.weight_e2v)
        # zeros(self.bias)

    def forward(self, x, data):
        r"""
        Args:
            x (Tensor): Node feature matrix :math:`\mathbf{X}`
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (Tensor, optional): Sparse hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
        """
        hyperedge_index = data.edge_index
        hyperedge_weight = None
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x = self.weight_v2e(x)

        x = data.D_v_beta.unsqueeze(-1) * x

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=data.D_e_beta_inv,
                             size=(num_nodes, num_edges))
        
        if self.nonlinear_inbetween:
            out = F.relu(out)
        
        # sanity check
        out = torch.squeeze(out, dim=1)
        
        out = self.weight_e2v(out)
        
        out = data.D_e_alpha.unsqueeze(-1) * out

        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=data.D_v_alpha_inv,
                             size=(num_edges, num_nodes))
        
        return out

    def message(self, x_j, norm_i):

        out = norm_i.view(-1, 1) * x_j

        return out

    def __repr__(self):
        return "{}({}, {}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.hidden_channels, self.out_channels)

class HNHN(nn.Module):

    @staticmethod
    def generate_norm(data, args):
        """
        :param H: hypergraph incidence matrix H
        :param variable_weight: whether the weight of hyperedge is variable
        :return: G
        """
    
        # Construct incidence matrix H of size (num_nodes, num_hyperedges) from edge_index = [V;E]
        edge_index = data.edge_index
        ones = torch.ones(data.edge_index.shape[1], device=edge_index.device)

        alpha = args.HNHN_alpha
        beta = args.HNHN_beta

        # the degree of the node
        DV = torch_scatter.scatter_add(ones, edge_index[0], dim=0)
        # the degree of the hyperedge
        DE = torch_scatter.scatter_add(ones, edge_index[1], dim=0)

        # alpha part
        D_e_alpha = DE ** alpha
        D_v_alpha = torch_scatter.scatter_add(DE[edge_index[1]], edge_index[0], dim=0)

        # beta part
        D_v_beta = DV ** beta
        D_e_beta = torch_scatter.scatter_add(DV[edge_index[0]], edge_index[1], dim=0)

        D_v_alpha_inv = 1.0 / D_v_alpha
        D_v_alpha_inv[D_v_alpha_inv == float("inf")] = 0

        D_e_beta_inv = 1.0 / D_e_beta
        D_e_beta_inv[D_e_beta_inv == float("inf")] = 0

        data.D_e_alpha = D_e_alpha.float()
        data.D_v_alpha_inv = D_v_alpha_inv.float()
        data.D_v_beta = D_v_beta.float()
        data.D_e_beta_inv = D_e_beta_inv.float()

        return data

    def __init__(self, num_features, num_classes, args):
        super(HNHN, self).__init__()

        self.num_layers = args.All_num_layers
        self.dropout = args.dropout
        
        self.convs = nn.ModuleList()
        # two cases
        if self.num_layers == 1:
            self.convs.append(HNHNConv(num_features, args.MLP_hidden, num_classes,
                                       nonlinear_inbetween=args.HNHN_nonlinear_inbetween))
        else:
            self.convs.append(HNHNConv(num_features, args.MLP_hidden, args.MLP_hidden,
                                       nonlinear_inbetween=args.HNHN_nonlinear_inbetween))
            for _ in range(self.num_layers - 2):
                self.convs.append(HNHNConv(args.MLP_hidden, args.MLP_hidden, args.MLP_hidden,
                                           nonlinear_inbetween=args.HNHN_nonlinear_inbetween))
            self.convs.append(HNHNConv(args.MLP_hidden, args.MLP_hidden, num_classes,
                                       nonlinear_inbetween=args.HNHN_nonlinear_inbetween))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):

        x = data.x
        
        if self.num_layers == 1:
            conv = self.convs[0]
            x = conv(x, data)
            # x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            for i, conv in enumerate(self.convs[:-1]):
                x = F.relu(conv(x, data))
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, data)

        return x