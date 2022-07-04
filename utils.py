import os, sys
import matplotlib.pyplot as plt

import numpy as np

import torch

class NodeClsEvaluator:

    def __init__(self):
        return

    def eval(self, y_true, y_pred):
        acc_list = []
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

        is_labeled = (~np.isnan(y_true)) & (~np.isinf(y_true)) # no nan and inf
        correct = (y_true[is_labeled] == y_pred[is_labeled])
        acc_list.append(float(np.sum(correct))/len(correct))

        return {'acc': sum(correct) / sum(is_labeled)}

class NodeRegEvaluator:

    def __init__(self):
        return

    def eval(self, y_true, y_pred):
        y_true = y_true.detach().cpu()
        y_pred = y_pred.detach().cpu()
        d = y_true - y_pred
        return {
            'mse': torch.mean(torch.square(d)).item(),
            'mae': torch.mean(torch.abs(d)).item(),
            'mape': torch.mean(torch.abs(d) / torch.abs(y_true)).item(),
        }

""" Adapted from https://github.com/snap-stanford/ogb/ """
class Logger:

    def __init__(self, runs, log_path=None):
        self.log_path = log_path
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, train_acc, valid_acc, test_acc):
        result = [train_acc, valid_acc, test_acc]
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def get_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            max_train = result[:, 0].max().item()
            max_test = result[:, 2].max().item()

            argmax = result[:, 1].argmax().item()
            train = result[argmax, 0].item()
            valid = result[argmax, 1].item()
            test = result[argmax, 2].item()
            return {'max_train': max_train, 'max_test': max_test,
                'train': train, 'valid': valid, 'test': test}
        else:
            keys = ['max_train', 'max_test', 'train', 'valid', 'test']

            best_results = []
            for r in range(len(self.results)):
                best_results.append([self.get_statistics(r)[k] for k in keys])

            ret_dict = {}
            best_result = torch.tensor(best_results)
            for i, k in enumerate(keys):
                ret_dict[k+'_mean'] = best_result[:, i].mean().item()
                ret_dict[k+'_std'] = best_result[:, i].std().item()

            return ret_dict

    def print_statistics(self, run=None):
        if run is not None:
            result = self.get_statistics(run)
            print(f"Run {run + 1:02d}:")
            print(f"Highest Train: {result['max_train']:.2f}")
            print(f"Highest Valid: {result['valid']:.2f}")
            print(f"  Final Train: {result['train']:.2f}")
            print(f"   Final Test: {result['test']:.2f}")
        else:
            result = self.get_statistics()
            print(f"All runs:")
            print(f"Highest Train: {result['max_train_mean']:.2f} ± {result['max_train_std']:.2f}")
            print(f"Highest Valid: {result['valid_mean']:.2f} ± {result['valid_std']:.2f}")
            print(f"  Final Train: {result['train_mean']:.2f} ± {result['train_std']:.2f}")
            print(f"   Final Test: {result['test_mean']:.2f} ± {result['test_std']:.2f}")

    def plot_result(self, run=None):
        plt.style.use('seaborn')
        if run is not None:
            result = 100 * torch.tensor(self.results).mean(0)
            x = torch.arange(result.shape[0])
            plt.figure()
            print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])
        else:
            result = 100 * torch.tensor(self.results[0])
            x = torch.arange(result.shape[0])
            plt.figure()
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])

""" Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
""" randomly splits label into train/valid/test splits """
def rand_train_test_idx(label, train_prop, valid_prop, balance=False):
    if not balance:
        n = label.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        perm = torch.randperm(n)

        train_idx = perm[:train_num]
        valid_idx = perm[train_num:train_num + valid_num]
        test_idx = perm[train_num + valid_num:]

        split_idx = {
            'train': train_idx,
            'valid': valid_idx,
            'test': test_idx
        }

    else:
        indices = []
        for i in range(label.max()+1):
            index = torch.where((label == i))[0].view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        percls_trn = int(train_prop/(label.max()+1)*len(label))
        val_lb = int(valid_prop*len(label))
        train_idx = torch.cat([ind[:percls_trn] for ind in indices], dim=0)
        rest_index = torch.cat([ind[percls_trn:] for ind in indices], dim=0)
        valid_idx = rest_index[:val_lb]
        test_idx = rest_index[val_lb:]

        split_idx = {
            'train': train_idx,
            'valid': valid_idx,
            'test': test_idx
        }

    return split_idx

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
