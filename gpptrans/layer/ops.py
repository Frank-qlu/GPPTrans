import torch
import torch.nn as nn
import numpy as np

class GraphUnet(nn.Module):
    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
    # """
    # :param ks: 表示pools层进行的节点采样率，数据类型为float型
    # """
        super(GraphUnet, self).__init__()
        self.ks = ks
        # 对应原文的encoder中的gcn
#         self.down_gcns = nn.ModuleList()
         # 对应原文的encoder中的gPool
        self.pools = nn.ModuleList()
        # 对应原文的decoder中的gcn
         # 创建底部的gcn
#         self.bottom_gcn = GCN(dim, dim, act, drop_p)
#         self.up_gcns = nn.ModuleList()
        # 对应原文的decoder中的gUnPool
        self.unpools = nn.ModuleList() 
#         self.decoder_gcn = GCN(dim*6, dim, act, drop_p)
        # ks的长度表示了gUNets的深度
        self.l_n = len(ks)
        # 构建l_n个子模块
        for i in range(self.l_n):
#             self.down_gcns.append(GCN(dim, dim, act, drop_p))
            self.pools.append(Pool(ks[i], dim, drop_p))
#             self.up_gcns.append(GCN(dim, dim, act, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))

    def forward(self, g, h):
    # """
    # : param g:邻接矩阵
    # : param h:特征矩阵
    # """
        adj_ms = []
        h_concat=[]
        indices_list = [] # 用于存储TopK的idx
        down_outs = [] # 存储经过encoder中gcn产生的特征
        hs = []
        org_h = h # 原始的特征信息
        org_g=g  #原始的图
        pool_g=[] #池化后的图
        pool_h=[] #池化后的特征
        decoder_g=[]
        # Encoder执行部分
        for i in range(self.l_n): #self.l_n:3
#             h = self.down_gcns[i](g, h) # h为encoder产生的特征
            adj_ms.append(g) # 存储输入第i层的邻接矩阵
            down_outs.append(h) # 存储第i层输出的特征矩阵
            # 经过图池化操作
            g, h, idx = self.pools[i](g, h) # idx存储了重要的TopK的节点信息
            indices_list.append(idx) # 存储第i层保留的idx的信息，用于decoder的信息还原
#             h = self.bottom_gcn(g, h)#编码GCN
            pool_g.append(g)
            pool_h.append(h)
            g,h=org_g,org_h
        # # bottom gcn部分
        
        # Decoder执行部分
        for i in range(self.l_n):
            g,h=pool_g[i],pool_h[i]
			# 由于decoder部分将编码后的邻接矩阵从小恢复到大
            up_idx = i
#             up_idx = self.l_n - i - 1
            # 分别取出与当前decoder block对应的邻接矩阵g，以及节点索引idx
            g, idx = adj_ms[up_idx], indices_list[up_idx]
            # 通过gUnPool操作恢复
            # 输出将增加邻接矩阵的维度
            g, h = self.unpools[i](g, h, down_outs[up_idx], idx)
            # h = self.up_gcns[i](g, h)#1875
            # h为第n_l-i层的特征输出
            # 将对应encoder block与decoder block进行skip connection
            # h = h.add(down_outs[up_idx])
            # # 存储经过跳跃连接之后的特征矩阵
            # hs.append(h)
            decoder_g.append(g)
            h_concat.append(h)
        # h = h.add(org_h) # 将原始的特征矩阵信息与经过所有block输出的特征矩阵的信息进行skip connection
        # hs.append(h)
#         hs=torch.concat(h_concat,dim=1)
#         hs=h_concat[0]+h_concat[1]+h_concat[-1]
        #hs = self.decoder_gcn(org_g, hs) #恢复到dim维度
        hs=h_concat[0]+h_concat[1]+h_concat[2]+h_concat[3]+h_concat[4]+h_concat[5]
        return hs # 输出最终的Embedding vector


class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, act, p):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

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
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return g, new_h


def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx


def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g


class Initializer(object):

    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)
