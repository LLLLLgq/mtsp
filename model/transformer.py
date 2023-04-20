import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mtsp.utils.util import check, init


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)


class Attention(nn.Module):

    def __init__(self, n_embed, n_head, n_nodes, masked=False):
        super(Attention, self).__init__()
        self.n_embed = n_embed
        self.n_head = n_head
        assert n_embed % n_head == 0
        self.n_nodes = n_nodes
        self.masked = masked

        self.query = init_(nn.Linear(n_embed, n_embed))
        self.key = init_(nn.Linear(n_embed, n_embed))
        self.value = init_(nn.Linear(n_embed, n_embed))
        self.proj = init_(nn.Linear(n_embed, n_embed))

    def forward(self, q, k, v, d=None):
        B, N, E = q.shape

        q = self.query(q).view(B, N, self.n_head, E // self.n_head).permute(0, 2, 1, 3)  # B, H, N, E//H
        k = self.key(k).view(B, N, self.n_head, E // self.n_head).permute(0, 2, 1, 3)  # B, H, N, E//H
        v = self.value(v).view(B, N, self.n_head, E // self.n_head).permute(0, 2, 1, 3)  # B, H, N, E//H

        # B, H, N, N
        att = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(k.size(-1)))

        if d is not None:
            # D: B, N, N   dist matrix
            # GraphDistAtt Y = Φ(D)*softmax(QK^T/sqrt(d)+Φ(D))V
            d = d.unsqueeze(1).repeat(1, self.n_head, 1, 1)
            att = att + d

        if self.masked:
            att = att.masked_fill(self.masked == 0, -1e9)
            att = F.softmax(att, dim=-1)
        else:
            att = F.softmax(att, dim=-1)

        if d is not None:
            att = att * d

        x = att @ v  # B, H, N, E//H
        x = x.permute(0, 2, 1, 3).contiguous().view(B, N, E)  # B, N, E

        x = self.proj(x)

        return x


class EncoderBlock(nn.Module):

    def __init__(self, n_embed, n_head, n_nodes, normalization="batch", masked=None):
        super(EncoderBlock, self).__init__()
        self.n_embed = n_embed
        self.n_head = n_head
        self.n_nodes = n_nodes
        self.masked = masked
        self.normalization = normalization
        self.att = Attention(n_embed, n_head, n_nodes, masked)

        if normalization == "layer":
            self.norm1 = nn.LayerNorm(n_embed)
            self.norm2 = nn.LayerNorm(n_embed)
        elif normalization == "batch":
            self.norm1 = nn.BatchNorm1d(n_embed)
            self.norm2 = nn.BatchNorm1d(n_embed)
        else:
            raise NotImplementedError

        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embed, n_embed * 4)),
            nn.GELU(),
            init_(nn.Linear(n_embed * 4, n_embed))
        )

    def forward(self, input):
        x, d = input
        # B N E
        if self.normalization == "layer":
            x = self.norm1(x + self.att(x, x, x, d))
            x = self.norm2(x + self.mlp(x))
        elif self.normalization == "batch":
            x = self.norm1((x + self.att(x, x, x, d)).view(-1, self.n_embed)).view(-1, self.n_nodes, self.n_embed)
            x = self.norm2((x + self.mlp(x)).view(-1, self.n_embed)).view(-1, self.n_nodes, self.n_embed)
        return x, d


class GraphEncoder(nn.Module):

    def __init__(self, n_blocks, n_embed, node_dim, n_head, n_nodes, normalization, masked=None):
        super(GraphEncoder, self).__init__()

        self.init_embed = init_(nn.Linear(node_dim, n_embed))

        self.encoder = nn.Sequential(*(
            EncoderBlock(n_embed, n_head, n_nodes, normalization, masked=masked) for _ in range(n_blocks)
        ))

        # #init
        # for name, param in self.named_parameters():
        #     stdv = 1. / math.sqrt(param.size(-1))
        #     param.data.uniform_(-stdv, stdv)

    def forward(self, x, d=None, mask=None):
        # x: B N D
        x = self.init_embed(x)
        # x: B N E
        x = self.encoder((x, d))
        # x: B N E
        x, _ = x
        graph_embedding = x.mean(dim=-2)
        return x, graph_embedding  # node embedding, graph embedding


if __name__ == '__main__':
    model = GraphEncoder(2, 128, 128, 8, 20, "batch")
    x = torch.randn(2, 20, 128)
    d = torch.randn(2, 20, 20)
    print(model(x, d))
