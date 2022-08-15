from re import L
import torch

import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing
from models.mlp import MLP

import numpy as np
import math 

import torch_scatter

class HyperND(nn.Module):
    def __init__(self, num_features, num_classes, args):

        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes

        self.p = args.HyperND_ord
        self.tol = args.HyperND_tol
        self.max_steps = args.HyperND_steps
        self.aggr = args.aggregate
        self.alpha = args.restart_alpha

        self.cache = None

        # self.lin = nn.Linear(num_features, num_classes)
        self.classifier = MLP(in_channels=num_features,
            hidden_channels=args.Classifier_hidden,
            out_channels=num_classes,
            num_layers=args.Classifier_num_layers,
            dropout=args.dropout,
            Normalization=args.normalization,
            InputNorm=False)

    def reset_parameters(self):
        self.cache = None
        self.classifier.reset_parameters()

    def diffusion(self, data):
        if self.cache is not None:
            return self.cache

        X = data.x
        # Y = nn.functional.one_hot(data.y)
        N = X.shape[-2]
        V, E = data.edge_index[0], data.edge_index[1]
        device = X.device

        ones = torch.ones(V.shape[0], device=device, dtype=torch.float32)
        D = torch_scatter.scatter_add(ones, V, dim=0)

        ones = torch.ones(E.shape[0], device=device, dtype=torch.float32)
        DE = torch_scatter.scatter_add(ones, E, dim=0)

        def rho(Z):
            return torch.pow(Z, self.p)

        def sigma(Z):
            Z = Z / DE[:, None]
            return torch.pow(Z, 1. / self.p)

        def V2E(Z):
            Z = Z / torch.pow(D, 0.5)[:, None]
            Xve = rho(Z[..., V, :]) # [nnz, C]
            Xe = torch_scatter.scatter(Xve, E, dim=-2, reduce=self.aggr) # [E, C]
            Xe = sigma(Xe)
            return Xe

        def E2V(Z):
            Xev = Z[..., E, :] # [nnz, C]
            Xv = torch_scatter.scatter(Xev, V, dim=-2, reduce=self.aggr, dim_size=N) # [N, C]
            Xv = Xv / torch.pow(D, 0.5)[:, None]
            return Xv

        def phi(Z):
            P = V2E(Z)
            P = torch.linalg.norm(P, dim=-1, ord=2.) ** 2.
            return 2. * torch.sqrt(P.sum())

        # U = torch.cat([X, Y], -1) + 1e-6
        U = X / phi(X)
        F = U
        for i in range(self.max_steps):
            F_last = F

            G = (1. - self.alpha) * E2V(V2E(F)) + self.alpha * U
            F = G / phi(G)


            d = torch.linalg.norm(F - F_last, ord='fro') / torch.linalg.norm(F, ord='fro')
            if d < self.tol:
                print(f'Interrupt hypergraph diffusion with {i} iterations and d={d}')
                break

        self.cache = F

        return F

    def forward(self, data):
        F = self.diffusion(data)
        return self.classifier(F)
        