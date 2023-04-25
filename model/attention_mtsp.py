import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import GraphEncoder


class Attention_MTSP(nn.Module):
    def __init__(self, args):
        super(Attention_MTSP, self).__init__()
        self.args = args
        self.problem = args.problem
        self.n_nodes = args.graph_size
        self.node_dim = args.node_dim
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads
        self.n_embed = args.n_embed
        self.normalization = args.normalization
        self.n_agents = args.n_agents
        self.GraphEncoder = GraphEncoder(self.n_layers, self.n_embed, self.node_dim, self.n_heads, self.n_nodes,
                                         self.normalization)
        assert self.n_embed % self.n_heads == 0
        self.dim = self.n_embed // self.n_heads

    def forward(self, x):
        state = self.problem.make_state(x)
        # x: [batch_size, n_nodes, node_dim]
        node_embedding, graph_embedding = self.GraphEncoder(x)  # [batch_size, n_nodes, n_embed], [batch_size, n_embed]
        B, N, E = node_embedding.shape
        agent_embedding = torch.zeros(B, self.n_agents, E).to(self.args.device)  # [batch_size, n_agents, n_embed]
        for i in range(self.n_agents):
            agent_embedding[:, i, :] = node_embedding[:, 0, :]
        step_context = torch.zeros(B, self.n_agents, E).to(self.args.device)  # [batch_size, n_agents, n_embed]
        step = 1

        while step < N:
            last_action, last_agent = state.get_current_state()  # [batch_size], [batch_size]
