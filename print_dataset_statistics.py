import os, sys

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
import torch_scatter
import torch_sparse

import pandas as pd
from tqdm import tqdm
import configargparse
import matplotlib.pyplot as plt

import datasets

if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    # Dataset specific arguments
    parser.add_argument('--dname', default='walmart-trips-100')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--raw_data_dir', type=str, required=True)
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--valid_prop', type=float, default=0.25)
    parser.add_argument('--feature_noise', default='1', type=str, help='std for synthetic feature noise')
    parser.add_argument('--normtype', default='all_one', choices=['all_one','deg_half_sym'])
    parser.add_argument('--add_self_loop', action='store_false')
    parser.add_argument('--exclude_self', action='store_true', help='whether the he contain self node or not')
    parser.add_argument('--cuda', default=0, type=int)

    args = parser.parse_args()

    device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')

    transform = torch_geometric.transforms.Compose([datasets.AddHypergraphSelfLoops()])
    data = datasets.HypergraphDataset(root=args.data_dir, name=args.dname, 
        path_to_download=args.raw_data_dir, feature_noise=args.feature_noise, transform=transform).data
    V, E = data.edge_index
    V, E = V.to(device), E.to(device)
    
    V = V - V.min()
    E = E - E.min()

    num_edges = E.max() - E.min() + 1
    num_verts = V.max() - V.min() + 1
    print('# nodes:', num_verts.item())
    print('# edges:', num_edges.item() - num_verts.item()) # minus self-loops

    print('# features:', data.num_features)
    print('# classes:', data.num_classes)

    ones = torch.ones(V.shape[0], device=device, dtype=torch.float32)
    V2E_deg = torch_scatter.scatter_add(ones, V, dim=0)
    plt.hist(V2E_deg.cpu().numpy())
    plt.title(f'vert-edge deg: {V2E_deg.float().mean().item():.3f}')
    plt.savefig(os.path.join(args.data_dir, 'node_deg_hist.jpg'))
    plt.close()
    print(f'vert-edge deg: {V2E_deg.float().mean().item():.3f}')

    E2V_deg = torch_scatter.scatter_add(ones, E, dim=0)
    plt.hist(E2V_deg.cpu().numpy())
    plt.title(f'edge-vert deg: {E2V_deg.float().mean().item():.3f}')
    plt.savefig(os.path.join(args.data_dir, 'edge_deg_hist.jpg'))
    plt.close()
    print(f'edge-vert deg: {E2V_deg.float().mean().item():.3f}')

    clique, value = torch_sparse.spspmm(torch.stack([V, E], 0), ones,
        torch.stack([E, V], 0), ones, num_verts, num_edges, num_verts, coalesced=True)
    clique, value = torch_sparse.coalesce(clique, value, num_verts, num_verts, op="add")
    V2V_degs = torch_scatter.scatter_add(torch.ones_like(clique[0]).float(), clique[0], dim=0)
    plt.hist(V2V_degs.cpu().numpy())
    plt.title(f'vert-vert deg: {V2V_degs.float().mean().item():.3f}')
    plt.savefig(os.path.join(args.data_dir, 'CE_deg_hist.jpg'))
    plt.close()
    print(f'vert-vert deg: {V2V_degs.float().mean().item():.3f}')


    y = data.y.to(device)
    classes = torch_scatter.scatter_add(torch.ones_like(y, dtype=torch.float32), y, dim=0)
    plt.bar(np.arange(data.num_classes), classes.cpu().numpy())
    plt.title(f'cls dist')
    plt.savefig(os.path.join(args.data_dir, 'cls_dist.jpg'))
    plt.close()

    y_src, y_dst = y[clique[0]], y[clique[1]]
    equals = torch_scatter.scatter_add((y_src == y_dst).float(), clique[0], dim=0)
    homophily = equals / V2V_degs
    homophily[V2V_degs == 0] = 0
    plt.title(f'homophily: {homophily.mean().item():.3f}')
    plt.hist(homophily.cpu().numpy())
    plt.savefig(os.path.join(args.data_dir, 'homophily.jpg'))
    plt.close()
    print(f'homophily: {homophily.mean().item():.3f}')
