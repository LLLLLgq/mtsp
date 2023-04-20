import math

import torch.nn as nn

from .transformer import GraphEncoder
import torch
import numpy as np
from mtsp.problem.tsp.problem_tsp import TSP


class AttentionModel(nn.Module):
    def __init__(self,
                 args):
        super(AttentionModel, self).__init__()
        self.n_embed = args.n_embed
        self.n_head = args.n_head
        self.node_dim = args.node_dim
        self.n_nodes = args.graph_size
        self.n_layers = args.n_layers
        self.problem = args.problem
        self.tanh_clipping = args.tanh_clipping
        self.decoder_mode = args.decoder_mode
        self.normalization = args.encoder_normalization
        self.GraphEncoder = GraphEncoder(self.n_layers, self.n_embed, self.node_dim, self.n_head, self.n_nodes, self.normalization)
        self.step_contex_proj = nn.Linear(2 * self.n_embed, self.n_embed)
        self.fixed_contex_proj = nn.Linear(self.n_embed, self.n_embed)
        self.qkv_proj = nn.Linear(self.n_embed, 3 * self.n_embed)
        self.mlp = nn.Linear(self.n_embed, self.n_embed)
        self.device = torch.device("cuda:0" if not args.no_cuda else "cpu")
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * self.n_embed).to(self.device))
        self.W_placeholder.data.uniform_(-1, 1)
        if args.baseline == 'critic':
            self.value_head = nn.Sequential(nn.Linear(args.n_embed, 4 * args.n_embed),
                                            nn.ReLU(),
                                            nn.Linear(4 * args.n_embed, 1))
        assert self.n_embed % self.n_head == 0, "embedding dimension must be divisible by number of heads"

    def forward(self, x):
        # x: [batch_size, n_nodes, node_dim]
        B, N, _ = x.shape
        state = self.problem.make_state(x)
        d = state.get_dist()
        node_embeddings, graph_embedding = self.GraphEncoder(x, d)  # [batch_size, n_nodes, n_embed], [batch_size, n_embed]
        self.graph_embedding = graph_embedding.detach()
        # print("node_embeddings", node_embeddings)
        k, v, logit_k = self.qkv_proj(node_embeddings).chunk(3, dim=-1)  # [batch_size, n_nodes, n_embed]
        k = k.unsqueeze(2).contiguous().view(B, N, self.n_head, self.n_embed // self.n_head).permute(0, 2, 1,
                                                                                                     3)  # [batch_size, n_head, n_nodes, n_embed/n_head]
        v = v.unsqueeze(2).contiguous().view(B, N, self.n_head, self.n_embed // self.n_head).permute(0, 2, 1,
                                                                                                     3)  # [batch_size, n_head, n_nodes, n_embed/n_head]
        dim = self.n_embed // self.n_head
        fixed_context_embedding = self.fixed_contex_proj(graph_embedding)  # [batch_size, n_embed]
        i = 0
        output_log_p = []
        sequence = []
        while i < N - 1:
            last_action, _ = state.get_current_node()
            last_action = last_action.unsqueeze(-1).expand(B, 1, self.n_embed)  # [batch_size, 1, n_embed]
            q = fixed_context_embedding + self.step_contex_proj(
                torch.cat((node_embeddings[:, 0, :], torch.gather(node_embeddings, 1, last_action).squeeze(1)),
                          dim=-1))  # [batch_size, n_embed]
            mask = state.get_mask()  # [batch_size, n_nodes]
            q = q.unsqueeze(1).contiguous().view(B, 1, self.n_head, self.n_embed // self.n_head).permute(0, 2, 1,
                                                                                                         3)  # [batch_size, n_head, 1, n_embed/n_head]
            attn = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(
                dim))  # [batch_size, n_head, 1, n_nodes]
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2),
                                    -math.inf)  # [batch_size, n_head, 1, n_nodes]
            attn = torch.softmax(attn, dim=-1)  # [batch_size, n_head, 1, n_nodes]

            context = (attn @ v).permute(0, 2, 1, 3).contiguous().view(B, 1, self.n_embed)  # [batch_size, 1, n_embed]
            x = self.mlp(context)  # [batch_size, 1, n_embed]
            logits = (x @ logit_k.transpose(-1, -2)) * (1.0 / math.sqrt(
                dim))  # [batch_size, 1, n_nodes]
            logits = torch.tanh(logits) * self.tanh_clipping  # [batch_size, 1, n_nodes]
            logits_ = logits.masked_fill(mask.unsqueeze(1), -math.inf).squeeze(1)  # [batch_size, n_nodes]
            log_p = torch.log_softmax(logits_, dim=-1)
            assert not torch.isnan(log_p).any()
            actions = self.select_actions(log_p.exp(), mode=self.decoder_mode)  # [batch_size]
            output_log_p.append(log_p)
            sequence.append(actions)
            state = state.update(actions)
            i += 1
        log_p = torch.stack(output_log_p, dim=1)  # [batch_size, n_step, n_nodes]

        sequence = torch.stack(sequence, dim=1)  # [batch_size, n_step]
        _log_p = torch.gather(log_p, dim=-1, index=sequence.unsqueeze(-1)).squeeze(-1)  # [batch_size, n_nodes]
        _log_p_ = _log_p.sum(dim=-1)
        cost = state.get_cost(sequence)  # [batch_size]
        return cost, _log_p_

    def select_actions(self, log_p, mode="greedy"):
        if mode == "greedy":
            return torch.argmax(log_p, dim=-1)
        elif mode == "sample":
            return torch.multinomial(log_p, 1).squeeze(-1)
        else:
            raise NotImplementedError

    def get_val(self):
        return self.value_head(self.graph_embedding).squeeze(-1)  # [batch_size]

    def set_decoder_mode(self, mode):
        self.decoder_mode = mode