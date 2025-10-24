import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_

from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


class GPPTansAttention(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, use_bias, dim_edge=None, use_virt_nodes=False):
        super().__init__()

        if out_dim % num_heads != 0:
            raise ValueError('hidden dimension is not dividable by the number of heads')
        self.out_dim = out_dim // num_heads
        self.num_heads = num_heads
        self.use_virt_nodes = use_virt_nodes
        self.use_bias = use_bias

        if dim_edge is None:
            dim_edge = in_dim

        self.Q = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(dim_edge, self.out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)

    def visualize_original_graph(self, edge_index, num_nodes):
        # 创建 NetworkX 图
        G = nx.Graph()
        for u, v in edge_index.t().tolist():
            G.add_edge(u, v)

        # 绘制图
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, font_size=10, font_weight='bold', edge_color='gray')
        plt.title("Original Graph")
        plt.savefig('Original Graph.pdf', format='pdf', dpi=1200)
        plt.show()

    def propagate_attention(self, batch, edge_index):
        # 可视化原始节点图
        #self.visualize_original_graph(batch.edge_index, batch.x.size(0))

        src = batch.K_h[edge_index[0].to(torch.long)]  # (num edges) x num_heads x out_dim
        dest = batch.Q_h[edge_index[1].to(torch.long)]  # (num edges) x num_heads x out_dim
        score = torch.mul(src, dest)  # element-wise multiplication

        # Scale scores by sqrt(d)
        score = score / np.sqrt(self.out_dim)

        # Use available edge features to modify the scores for edges
        # score = torch.mul(score, batch.E)  # (num real edges) x num_heads x out_dim
        score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))  # (num real edges) x num_heads x 1

        # Apply attention score to each source node to create edge messages
        msg = batch.V_h[edge_index[0].to(torch.long)] * score  # (num real edges) x num_heads x out_dim
        # Add-up real msgs in destination nodes as given by batch.edge_index[1]
        batch.wV = torch.zeros_like(batch.V_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, edge_index[1], dim=0, out=batch.wV, reduce='add')

        # Compute attention normalization coefficient
        batch.Z = score.new_zeros(batch.V_h.size(0), self.num_heads, 1)  # (num nodes in batch) x num_heads x 1
        scatter(score, edge_index[1], dim=0, out=batch.Z, reduce='add')

        # 提取注意力权重（这里只取第一个头的注意力权重）
        attention_weights = score.mean(dim=1).squeeze().detach().cpu().numpy()  # 挤压掉多余的维度
        # 确保注意力权重的形状与边的数量一致
        #assert attention_weights.shape[0] == batch.edge_index.size(1), "Attention weights and edge indices do not match"

        # 创建 N x N 的注意力矩阵
        num_nodes = batch.x.size(0)
        attention_matrix = np.zeros((num_nodes, num_nodes))

        # 填充注意力矩阵
        for i, (u, v) in enumerate(batch.edge_index.t().tolist()):
            attention_matrix[u, v] = attention_weights[i]

        # 计算每个节点对自身的注意力权重
        for i in range(num_nodes):
            attention_matrix[i, i] = 1.0  # 或者使用其他方法计算自注意力权重

        # # 使用 Seaborn 绘制热力图
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(attention_matrix, cmap="Blues", linewidths=.5, cbar_kws={"label": "Attention Weight"})
        # plt.title("GAT Attention Heatmap")
        # plt.xlabel("Node Index")
        # plt.ylabel("Node Index")

        # # 保存为矢量图
        # plt.savefig('gat_attention_heatmap.svg', format='svg', dpi=1200)
        # plt.savefig('gat_attention_heatmap.pdf', format='pdf', dpi=1200)

        # # 显示图
        # plt.show()
        assert 1

    def forward(self, batch):
        edge_index = batch.edge_index
        h = batch.x
        num_node = batch.batch.shape[0]

        # Add <CLS> token to the input
        cls_token = torch.zeros(1, h.size(1)).to(h.device)
        h = torch.cat([cls_token, h], dim=0)

        # Update edge index to consider <CLS> token
        # Add edges from <CLS> token to all other nodes
        cls_to_nodes = torch.stack([torch.zeros(num_node, dtype=torch.long, device=h.device),
                                    torch.arange(1, num_node + 1, dtype=torch.long, device=h.device)], dim=0)
        edge_index = torch.cat([cls_to_nodes, edge_index + 1], dim=1)

        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(batch, edge_index)

        h_out = batch.wV / (batch.Z + 1e-6)

        h_out = h_out.view(-1, self.out_dim * self.num_heads)

        batch.virt_h = h_out[num_node + 1:]
        h_out = h_out[1:num_node + 1]

        return h_out


register_layer('GPPTans', GPPTansAttention)

def get_activation(activation):
    if activation == 'relu':
        return 2, nn.ReLU()
    elif activation == 'gelu':
        return 2, nn.GELU()
    elif activation == 'silu':
        return 2, nn.SiLU()
    elif activation == 'glu':
        return 1, nn.GLU()
    else:
        raise ValueError(f'activation function {activation} is not valid!')


class GPPTansFullLayer(nn.Module):
    """GPPTans attention + FFN
    """

    def __init__(self, in_dim, out_dim, num_heads,
                 dropout=0.0,
                 dim_edge=None,
                 layer_norm=False, batch_norm=True,
                 activation='relu',
                 residual=True, use_bias=False, use_virt_nodes=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.attention = GPPTansAttention(in_dim, out_dim, num_heads,
                                          use_bias=use_bias, 
                                          dim_edge=dim_edge,
                                          use_virt_nodes=use_virt_nodes)

        self.O_h = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        factor, self.activation_fn = get_activation(activation=activation)
        self.FFN_h_layer2 = nn.Linear(out_dim * factor, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        # multi-head attention out
        h_attn_out = self.attention(batch)

        # Concat multi-head outputs
        h = h_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)

        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = self.activation_fn(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        return batch

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels, self.num_heads, self.residual)


register_layer('GPPTansLayer', GPPTansFullLayer)