# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# import os

# class GraphUnet(nn.Module):
#     def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
#         """
#         :param ks: list of pooling ratios (floats)
#         :param in_dim, out_dim, dim: feature dims
#         :param act: activation function
#         :param drop_p: dropout rate
#         """
#         super(GraphUnet, self).__init__()
#         self.ks = ks
#         self.pools = nn.ModuleList()
#         self.unpools = nn.ModuleList()
#         self.l_n = len(ks)
#         for i in range(self.l_n):
#             self.pools.append(Pool(ks[i], dim, drop_p))
#             self.unpools.append(Unpool(dim, dim, drop_p))

#     def forward(self, g, h, labels=None, return_all_scales=False, tsne_vis=False, save_dir="./tsne_vis"):
#         """
#         g: adjacency matrix (dense), shape [N, N]
#         h: node features, shape [N, D]
#         labels: node labels, shape [N] (optional, for t-SNE coloring)
#         return_all_scales: if True, return (hs, pool_h_list, pool_g_list, indices_list)
#         tsne_vis: whether to run t-SNE visualization at each scale
#         """
#         adj_ms = []
#         indices_list = []
#         down_outs = []
#         pool_g = []
#         pool_h = []
#         org_h = h
#         org_g = g

#         N = org_h.shape[0]
#         D = org_h.shape[1]

#         if tsne_vis and not os.path.exists(save_dir):
#             os.makedirs(save_dir, exist_ok=True)

#         # Encoder: pooling at multiple scales
#         for i in range(self.l_n):
#             adj_ms.append(org_g.clone())
#             down_outs.append(org_h.clone())

#             g_p, h_p, idx = self.pools[i](org_g, org_h)
#             g_p = g_p.contiguous()
#             h_p = h_p.contiguous()
#             idx = idx.contiguous()

#             indices_list.append(idx)
#             pool_g.append(g_p)
#             pool_h.append(h_p)

#             # === 新增：每个池化尺度的 t-SNE 可视化 ===
#             if tsne_vis and h_p.shape[0] > 2:
#                 label_subset = labels[idx].cpu().numpy() if labels is not None else None
#                 self.visualize_tsne(
#                     h_p.detach().cpu(),
#                     labels=label_subset,
#                     title=f"Pooling Scale {self.ks[i]}",
#                     save_path=os.path.join(save_dir, f"scale_{i}_tsne.png")
#                 )

#         # Decoder-like aggregation: unpool each pooled scale back to original node count
#         h_unpooled_list = []
#         for i in range(self.l_n):
#             g_small = pool_g[i]
#             h_small = pool_h[i]
#             g_big = adj_ms[i]
#             idx = indices_list[i]

#             g_up, h_up = self.unpools[i](g_big, h_small, down_outs[i], idx)

#             # Safety: ensure [N, D]
#             if h_up.shape[0] != N or h_up.shape[1] != D:
#                 new_h_up = h_up.new_zeros((N, D))
#                 try:
#                     if idx.numel() > 0 and idx.max().item() < N:
#                         new_h_up[idx] = h_up
#                     else:
#                         min_rows = min(h_up.shape[0], N)
#                         new_h_up[:min_rows] = h_up[:min_rows]
#                 except Exception:
#                     min_rows = min(h_up.shape[0], N)
#                     new_h_up[:min_rows] = h_up[:min_rows]
#                 h_up = new_h_up

#             h_unpooled_list.append(h_up.contiguous().float())

#         # Aggregate unpooled embeddings
#         hs = torch.zeros((N, D), dtype=h_unpooled_list[0].dtype, device=h_unpooled_list[0].device)
#         for hu in h_unpooled_list:
#             if hu.shape != hs.shape:
#                 new_hu = torch.zeros_like(hs)
#                 min_rows = min(new_hu.shape[0], hu.shape[0])
#                 min_cols = min(new_hu.shape[1], hu.shape[1])
#                 new_hu[:min_rows, :min_cols] = hu[:min_rows, :min_cols]
#                 hu = new_hu
#             hs = hs + hu

#         # === 新增: t-SNE 可视化最终输出 ===
#         if tsne_vis and hs.shape[0] > 2:
#             labels_np = labels.cpu().numpy() if labels is not None else None
#             self.visualize_tsne(
#                 hs.detach().cpu(),
#                 labels=labels_np,
#                 title="GraphUnet Output Embeddings",
#                 save_path=os.path.join(save_dir, "final_tsne.png")
#             )

#         if return_all_scales:
#             return hs, pool_h, pool_g, indices_list
#         else:
#             return hs

#     # === 新增: t-SNE 可视化方法 ===
#     @staticmethod
#     def visualize_tsne(features, labels=None, title="t-SNE", save_path="./tsne.png"):
#         """
#         features: [N, D] 节点特征
#         labels: [N] 节点标签，可用于颜色区分
#         """
#         try:
#             if features.shape[0] > 2000:
#                 idx = np.random.choice(features.shape[0], 2000, replace=False)
#                 features = features[idx]
#                 if labels is not None:
#                     labels = labels[idx]

#             tsne = TSNE(
#                 n_components=2,
#                 perplexity=min(30, max(5, features.shape[0] // 3)),
#                 n_iter=1000,
#                 learning_rate=200,
#                 init="pca",
#                 random_state=42
#             )
#             emb_2d = tsne.fit_transform(features.numpy())
#             plt.figure(figsize=(5, 5))
            
#             if labels is not None:
#                 num_classes = len(np.unique(labels))
#                 cmap = plt.get_cmap("tab10", num_classes)
#                 scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap=cmap, s=10, alpha=0.7)
#                 plt.colorbar(scatter, ticks=range(num_classes))
#             else:
#                 plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=10, alpha=0.7)

#             plt.title(title)
#             plt.tight_layout()
#             plt.savefig(save_path, dpi=300)
#             plt.close()
#             print(f"[t-SNE] Saved to {save_path}")
#         except Exception as e:
#             print(f"[t-SNE error] {e}")


# class GCN(nn.Module):
#     def __init__(self, in_dim, out_dim, act, p):
#         super(GCN, self).__init__()
#         self.proj = nn.Linear(in_dim, out_dim)
#         self.act = act
#         self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

#     def forward(self, g, h):
#         h = self.drop(h)
#         h = torch.matmul(g, h)
#         h = self.proj(h)
#         h = self.act(h)
#         return h


# class Pool(nn.Module):
#     def __init__(self, k, in_dim, p):
#         super(Pool, self).__init__()
#         self.k = k
#         self.sigmoid = nn.Sigmoid()
#         self.proj = nn.Linear(in_dim, 1)
#         self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

#     def forward(self, g, h):
#         Z = self.drop(h)
#         weights = self.proj(Z).squeeze()
#         scores = self.sigmoid(weights)
#         return top_k_graph(scores, g, h, self.k)


# class Unpool(nn.Module):
#     def __init__(self, *args):
#         super(Unpool, self).__init__()

#     def forward(self, g, h, pre_h, idx):
#         new_h = pre_h.new_zeros([pre_h.shape[0], h.shape[1]])
#         new_h[idx] = h
#         return g, new_h


# def top_k_graph(scores, g, h, k):
#     num_nodes = g.shape[0]
#     k_num = max(2, int(k * num_nodes))
#     values, idx = torch.topk(scores, k_num)
#     new_h = h[idx, :]
#     values = torch.unsqueeze(values, -1)
#     new_h = torch.mul(new_h, values)

#     un_g = g.bool().float()
#     un_g = torch.matmul(un_g, un_g).bool().float()
#     un_g = un_g[idx, :][:, idx]
#     g = norm_g(un_g)
#     return g, new_h, idx


# def norm_g(g):
#     degrees = torch.sum(g, 1)
#     degrees = torch.where(degrees == 0, torch.ones_like(degrees), degrees)
#     g = g / degrees.unsqueeze(1)
#     return g
# graphunet_with_scales.py (modified)
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import os


# ========================
#  Graph U-Net 模块
# ========================
class GraphUnet(nn.Module):
    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
        """
        ks: list of pooling ratios (e.g., [0.9,0.8,...])
        dim: hidden dimension
        act: activation function
        drop_p: dropout probability
        """
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.l_n = len(ks)
        self.pools = nn.ModuleList([Pool(k, dim, drop_p) for k in ks])
        self.unpools = nn.ModuleList([Unpool(dim, dim, drop_p) for _ in ks])

    def forward(self, g, h, labels=None, return_all_scales=False,
                tsne_vis=False, save_dir="./tsne_scales"):
        """
        g: adjacency matrix [N,N]
        h: node features [N,D]
        labels: node labels [N] (optional)
        return_all_scales: bool
        tsne_vis: whether to visualize
        """
        if tsne_vis and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        N, D = h.shape
        adj_ms, down_outs, indices_list, pool_g, pool_h = [], [], [], [], []
        cluster_labels_list, node_idx_list = [], []

        org_h, org_g = h, g

        # === Encoder 多尺度池化 ===
        for i in range(self.l_n):
            adj_ms.append(org_g.clone())
            down_outs.append(org_h.clone())

            g_p, h_p, idx = self.pools[i](org_g, org_h)
            g_p, h_p, idx = g_p.contiguous(), h_p.contiguous(), idx.contiguous()

            indices_list.append(idx)
            pool_g.append(g_p)
            pool_h.append(h_p)
            node_idx_list.append(idx.detach().cpu().numpy())

            # --- 聚类 ---
            try:
                n_clusters = max(2, min(10, h_p.shape[0] // 3))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(h_p.detach().cpu().numpy())
                cluster_labels_list.append(cluster_labels)
            except Exception as e:
                print(f"[KMeans error at scale {i}] {e}")
                cluster_labels_list.append(None)

            # --- t-SNE 可视化 ---
            if tsne_vis and h_p.shape[0] > 2:
                label_subset = labels[idx].cpu().numpy() if labels is not None else None
                # self.visualize_tsne(
                #     h_p.detach().cpu(),
                #     labels=label_subset if label_subset is not None else cluster_labels,
                #     title=f"Scale {self.ks[i]}",
                #     save_path=os.path.join(save_dir, f"scale_{i}_tsne.png")
                # )

            org_g, org_h = g_p, h_p

        # # === 尺度间一致性计算 ===
        # print("\n==== Multi-scale Consistency Metrics ====")
        for i in range(len(cluster_labels_list) - 1):
            c1, c2 = cluster_labels_list[i], cluster_labels_list[i + 1]
            idx1, idx2 = node_idx_list[i], node_idx_list[i + 1]

            # --- ARI (聚类一致性) ---
            ari = None
            if c1 is not None and c2 is not None and len(c1) > 1 and len(c2) > 1:
                n = min(len(c1), len(c2))
                try:
                    ari = adjusted_rand_score(c1[:n], c2[:n])
                except Exception:
                    ari = None

            # --- Jaccard (节点集合重叠率) ---
            try:
                jacc = len(set(idx1) & set(idx2)) / len(set(idx1) | set(idx2))
            except Exception:
                jacc = None

            # print(f"Scale {i} -> {i+1}: ARI={ari}, Jaccard={jacc}")

        # === Decoder 上采样阶段 ===
        h_unpooled_list = []
        for i in range(self.l_n):
            g_small, h_small, g_big, idx = pool_g[i], pool_h[i], adj_ms[i], indices_list[i]
            g_up, h_up = self.unpools[i](g_big, h_small, down_outs[i], idx)

            # 安全补齐维度
            if h_up.shape[0] != N or h_up.shape[1] != D:
                new_h_up = h_up.new_zeros((N, D))
                min_rows, min_cols = min(h_up.shape[0], N), min(h_up.shape[1], D)
                new_h_up[:min_rows, :min_cols] = h_up[:min_rows, :min_cols]
                h_up = new_h_up
            h_unpooled_list.append(h_up.contiguous().float())

        # 聚合所有尺度上采样后的表示
        hs = torch.zeros((N, D), dtype=h_unpooled_list[0].dtype, device=h_unpooled_list[0].device)
        for hu in h_unpooled_list:
            hs += hu

        # === 最终 t-SNE 可视化 ===
        if tsne_vis and hs.shape[0] > 2:
            labels_np = labels.cpu().numpy() if labels is not None else None
            # self.visualize_tsne(
            #     hs.detach().cpu(),
            #     labels=labels_np,
            #     title="Final Graph Representation",
            #     save_path=os.path.join(save_dir, "final_tsne.png")
            # )

        if return_all_scales:
            return hs, pool_h, pool_g, indices_list
        else:
            return hs

    # =======================
    # 可视化函数
    # =======================
    # @staticmethod
    # def visualize_tsne(features, labels=None, title="t-SNE", save_path="./tsne.png"):
    #     try:
    #         if features.shape[0] > 2000:
    #             idx = np.random.choice(features.shape[0], 2000, replace=False)
    #             features = features[idx]
    #             if labels is not None:
    #                 labels = labels[idx]

    #         tsne = TSNE(
    #             n_components=2,
    #             perplexity=min(30, max(5, features.shape[0] // 3)),
    #             n_iter=1000,
    #             learning_rate=200,
    #             init="pca",
    #             random_state=42
    #         )
    #         emb_2d = tsne.fit_transform(features.numpy())

    #         plt.figure(figsize=(5, 5))
    #         if labels is not None:
    #             num_classes = len(np.unique(labels))
    #             cmap = plt.get_cmap("tab10", num_classes)
    #             scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1],
    #                                   c=labels, cmap=cmap, s=10, alpha=0.7)
    #             plt.colorbar(scatter, ticks=range(num_classes))
    #         else:
    #             plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=10, alpha=0.7)

    #         plt.title(title)
    #         plt.tight_layout()
    #         plt.savefig(save_path, dpi=300)
    #         plt.close()
    #         print(f"[t-SNE] Saved to {save_path}")
    #     except Exception as e:
    #         print(f"[t-SNE error] {e}")


# ========================
#  辅助层定义
# ========================
class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, act, p):
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


# ========================
#  工具函数
# ========================
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