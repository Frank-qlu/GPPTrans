import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

class GraphPyramidPooling(nn.Module):
    def __init__(self, ks, in_dim, out_dim, dim, act=nn.ReLU(), drop_p=0.0):
        super(GraphPyramidPooling, self).__init__()
        self.ks = ks
        self.l_n = len(ks)
        self.pools = nn.ModuleList([Pool(k, dim, drop_p) for k in ks])
        self.unpools = nn.ModuleList([Unpool(dim, dim, drop_p) for _ in ks])
        self.act = act

    def forward(self, g, h, labels=None, return_all_scales=False, tsne_vis=False, save_dir="./tsne_scales"):
        if tsne_vis and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        N, D = h.shape
        adj_ms, down_outs, indices_list, pool_g, pool_h = [], [], [], [], []
        cluster_labels_list, node_idx_list = [], []

        org_h, org_g = h, g

        for i in range(self.l_n):
            adj_ms.append(org_g.clone())
            down_outs.append(org_h.clone())

            g_p, h_p, idx = self.pools[i](org_g, org_h)
            g_p, h_p, idx = g_p.contiguous(), h_p.contiguous(), idx.contiguous()

            indices_list.append(idx)
            pool_g.append(g_p)
            pool_h.append(h_p)
            node_idx_list.append(idx.detach().cpu().numpy())

            org_g, org_h = g_p, h_p
        h_unpooled_list = []
        for i in range(self.l_n):
            g_small, h_small, g_big, idx = pool_g[i], pool_h[i], adj_ms[i], indices_list[i]
            g_up, h_up = self.unpools[i](g_big, h_small, down_outs[i], idx)

            if h_up.shape[0] != N or h_up.shape[1] != D:
                new_h_up = h_up.new_zeros((N, D))
                min_rows, min_cols = min(h_up.shape[0], N), min(h_up.shape[1], D)
                new_h_up[:min_rows, :min_cols] = h_up[:min_rows, :min_cols]
                h_up = new_h_up
            h_unpooled_list.append(h_up.contiguous().float())
        hs = torch.zeros((N, D), dtype=h_unpooled_list[0].dtype, device=h_unpooled_list[0].device)
        for hu in h_unpooled_list:
            hs += hu
        return (hs, pool_h, pool_g, indices_list) if return_all_scales else hs

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.ReLU(), p=0.0):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        h = self.drop(h)
        h = torch.matmul(g, h)
        h = self.proj(h)
        h = self.act(h)
        return h


class Pool(nn.Module):
    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)


class Unpool(nn.Module):
    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        new_h = pre_h.new_zeros([pre_h.shape[0], h.shape[1]])
        new_h[idx] = h
        return g, new_h

def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]
    k_num = max(2, int(k * num_nodes))
    values, idx = torch.topk(scores, k_num)
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)

    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :][:, idx]
    g = norm_g(un_g)
    return g, new_h, idx


def norm_g(g):
    degrees = torch.sum(g, 1)
    degrees = torch.where(degrees == 0, torch.ones_like(degrees), degrees)
    g = g / degrees.unsqueeze(1)
    return g
