import math
import random
import time

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .transformer import Decoder as ClassifyDecoder
from .transformer import Encoder as GraphEncoder


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
        self.m_problem = args.m_problem
        self.tanh_clipping = args.tanh_clipping
        self.decoder_mode = args.decoder_mode
        self.epsilon = args.epsilon
        self.allocate_mode = args.allocate_mode
        # self.allocate_epsilon = args.allocate_epsilon
        self.allocate_order = args.allocate_order
        self.classify_mode = args.classify_mode
        self.normalization = args.encoder_normalization
        self.dec_actor = args.dec_actor
        self.device = torch.device("cuda:0" if not args.no_cuda else "cpu")
        self.GraphEncoder = GraphEncoder(self.n_layers, self.n_embed, self.node_dim, self.n_head, self.n_nodes,
                                         self.normalization)
        self.freezeGraphEncoder = None
        self.step_contex_proj = nn.Linear(2 * self.n_embed, self.n_embed)
        self.fixed_contex_proj = nn.Linear(self.n_embed, self.n_embed)
        self.qkv_proj = nn.Linear(self.n_embed, 3 * self.n_embed)
        self.mlp = nn.Linear(self.n_embed, self.n_embed)

        self.n_agents = args.n_agents
        self.divide_step_context_proj = nn.Linear(2 * self.n_embed, self.n_embed)
        self.divide_fixed_context_proj = nn.Linear(self.n_embed, self.n_embed)
        self.divide_qkv_proj = nn.Linear(self.n_embed, 3 * self.n_embed)
        self.divide_mlp = nn.Linear(self.n_embed, self.n_embed)

        self.classifier = ClassifyDecoder(self.n_layers, self.n_embed, self.node_dim, self.n_head, self.n_nodes,
                                          self.n_agents, self.normalization, self.dec_actor)

        if args.baseline == 'critic':
            self.value_head = nn.Sequential(nn.Linear(args.n_embed, 4 * args.n_embed),
                                            nn.ReLU(),
                                            nn.Linear(4 * args.n_embed, 1))
        assert self.n_embed % self.n_head == 0, "embedding dimension must be divisible by number of heads"
        self.dim = self.n_embed // self.n_head
        assert self.n_nodes >= self.n_agents, "graph size must be greater than or equal to the number of agents"

    def get_tsp_solution(self, x):
        # x: [batch_size, n_nodes, node_dim]
        B, N, _ = x.shape
        state = self.problem.make_state(x)
        d = state.get_dist()
        if self.freezeGraphEncoder is not None:
            node_embeddings, graph_embedding = self.freezeGraphEncoder(x, d)  # [batch_size, n_nodes, n_embed], [batch_size, n_embed]
        else:
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
        elif mode == "epsilon greedy":
            if random.random() < self.epsilon:
                return torch.multinomial(log_p, 1).squeeze(-1)
            else:
                return torch.argmax(log_p, dim=-1)
        else:
            raise NotImplementedError

    def get_val(self):
        return self.value_head(self.graph_embedding).squeeze(-1)  # [batch_size]

    def set_tsp_decoder_mode(self, mode):
        self.decoder_mode = mode

    def classify_node(self, x, model='attention'):
        # x: [batch_size, n_nodes, node_dim]
        st = time.time()
        B, N, _ = x.shape
        state = self.problem.make_state(x)
        enc, _ = self.GraphEncoder(x)  # [batch_size, n_nodes, n_embed]
        d = state.get_dist()
        # distance-aware node embedding
        prob = self.classifier(x, enc, d)[:, 1:, :]  # [batch_size, n_nodes-1, n_agents]
        if self.classify_mode == 'greedy':
            actions = torch.argmax(prob, dim=-1)  # [batch_size, n_nodes-1]
        elif self.classify_mode == 'sample':

            distri = Categorical(probs=prob.exp())
            actions = distri.sample()  # [batch_size, n_nodes-1]
        else:
            raise NotImplementedError
        log_p = prob.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)  # [batch_size, n_nodes-1]
        assignment = torch.zeros(B, self.n_agents, N, dtype=torch.int64).to(x.device)
        assignment[:,:,1:] = torch.scatter(assignment[:,:,1:], 1, actions.unsqueeze(1), torch.arange(1, N).unsqueeze(0).unsqueeze(0).repeat(B, self.n_agents, 1).to(x.device))
        return assignment, log_p.sum(dim=-1)  # [batch_size,n_agents, n_nodes], [batch_size]

    def divide_nodes(self, x):
        # x: [batch_size, n_nodes, node_dim]
        assert self.n_agents > 1, "n_agents should be larger than 1"
        B, N, _ = x.shape
        state = self.problem.make_state(x)
        d = state.get_dist()
        node_embeddings, graph_embedding = self.GraphEncoder(x,
                                                             d)  # [batch_size, n_nodes, n_embed], [batch_size, n_embed]
        agent_embedding = torch.zeros(B, self.n_agents, self.n_embed, dtype=torch.float32).to(
            x.device)  # [batch_size, n_agents, n_embed]
        step_context_embedding = torch.zeros(B, self.n_agents, 2 * self.n_embed, dtype=torch.float32).to(
            x.device)  # [batch_size, n_agents, n_embed]
        assignment = torch.zeros(B, self.n_agents, self.n_nodes, dtype=torch.int64).to(x.device)
        count = torch.ones((B, self.n_agents), dtype=torch.int)
        k, v, logit_k = self.divide_qkv_proj(node_embeddings).chunk(3, dim=-1)  # [batch_size, n_nodes, n_embed]
        k = k.unsqueeze(1).contiguous().view(B, N, self.n_head, self.n_embed // self.n_head).permute(0, 2, 1,
                                                                                                     3)  # [batch_size, n_head, n_nodes, n_embed/n_head]
        v = v.unsqueeze(1).contiguous().view(B, N, self.n_head, self.n_embed // self.n_head).permute(0, 2, 1,
                                                                                                     3)  # [batch_size, n_head, n_nodes, n_embed/n_head]

        for i in range(self.n_agents):
            agent_embedding[:, i, :] = node_embeddings[:, 0, :]  # [batch_size, n_embed]
        step = 1
        output_log = []
        while step < N:
            mask = state.get_mask()  # [batch_size, n_nodes]
            agents_embedding_avg = agent_embedding.sum(dim=1) * (1.0 / (self.n_agents - 1))  # [batch_size, n_embed]
            for i in range(self.n_agents):
                step_context_embedding[:, i, :] = torch.cat((agent_embedding[:, i, :],
                                                             agents_embedding_avg - agent_embedding[:, i, :] * (
                                                                     1.0 / (self.n_agents - 1)))
                                                            , dim=-1)  # [batch_size, 2*n_embed]
            context_q = self.fixed_contex_proj(graph_embedding).unsqueeze(1) + self.step_contex_proj(
                step_context_embedding)  # [batch_size, n_agents, n_embed]
            context_q = context_q.unsqueeze(2).contiguous().view(B, self.n_agents, self.n_head,
                                                                 self.n_embed // self.n_head).permute(0, 2, 1,
                                                                                                      3)  # [batch_size, n_head, n_agents, n_embed/n_head]
            attn = (context_q @ k.transpose(-1, -2)) * (
                    1.0 / math.sqrt(self.dim))  # [batch_size, n_head, n_agents, n_nodes]
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2),
                                    -math.inf)
            attn = torch.softmax(attn, dim=-1)  # [batch_size, n_head, n_agents, n_nodes]
            context = (attn @ v).permute(0, 2, 1, 3).contiguous().view(B, self.n_agents,
                                                                       self.n_embed)  # [batch_size, n_agents, n_embed]
            logit = (context @ logit_k.transpose(-1, -2)) * (
                    1.0 / math.sqrt(self.dim))  # [batch_size, n_agents, n_nodes]
            logit = logit.masked_fill(mask.unsqueeze(1), -1e9)  # [batch_size, n_agents, n_nodes]
            allocated_node, allocated_agent, log_p = self.allocate_node(logit, self.allocate_mode,
                                                                        self.allocate_order)  # [batch_size] [batch_size] [batch_size]
            # log_p = torch.log_softmax(logit, dim=-1)  # [batch_size, n_agents, n_nodes]
            output_log.append(log_p)
            state = state.update(allocated_node)
            # update agent embedding with average
            print('allocated_agent:{}, allocated_node:{}'.format(allocated_agent, allocated_node))
            assignment[:, :, step] = assignment[:, :, step].scatter(1, allocated_agent.unsqueeze(-1),
                                                                    allocated_node.unsqueeze(
                                                                        -1))  # [batch_size, n_agents, n_step]
            count = count.scatter_add(1, allocated_agent.unsqueeze(-1),
                                      torch.ones_like(allocated_agent, dtype=torch.int).unsqueeze(
                                          -1))  # [batch_size, n_agents]
            agent_embedding = agent_embedding * (
                    1.0 - 1.0 / count.unsqueeze(-1).repeat(1, 1, self.n_embed))  # [batch_size, n_agents, n_embed]
            agent_embedding = agent_embedding.scatter_add(1, allocated_agent.unsqueeze(-1).unsqueeze(-1).repeat(1, 1,
                                                                                                                self.n_embed),
                                                          node_embeddings.gather(1,
                                                                                 allocated_node.unsqueeze(-1).unsqueeze(
                                                                                     -1).repeat(1, 1, self.n_embed)) * (
                                                                  1.0 / count.unsqueeze(-1).repeat(1, 1,
                                                                                                   self.n_embed)))  # [batch_size, n_agents, n_embed]

            step += 1
        log_p = torch.stack(output_log, dim=1).sum(-1)  # [batch_size]

        return assignment, log_p

    def allocate_node(self, logit, mode, order="equally"):
        # logit: [batch_size, n_agents, n_nodes]
        # mode: "greedy", "sample", "epsilon greedy"
        # order: "equally", "node first", "agent first"
        allocated_agent = None
        allocated_node = None
        if order == "equally":
            pro = logit.view(logit.shape[0], -1)  # [batch_size, n_agents * n_nodes]
            log_p = torch.log_softmax(pro, dim=-1)  # [batch_size, n_agents * n_nodes]
            category = Categorical(probs=log_p.exp())  # bug
            if mode == "greedy":
                allocated = category.probs.argmax(dim=-1)
            elif mode == "sample":
                allocated = category.sample()
            elif mode == "epsilon greedy":
                if random.random() < self.allocate_epsilon:
                    allocated = category.sample()
                else:
                    allocated = category.probs.argmax(dim=-1)
            else:
                raise NotImplementedError
            return allocated % self.n_nodes, allocated // self.n_nodes, log_p.gather(1,
                                                                                     allocated.unsqueeze(-1)).squeeze(
                -1)  # [batch_size], [batch_size], [batch_size]

        elif order == "agent first":
            agent_intention = torch.softmax(logit.sum(dim=-1), dim=-1)  # [batch_size, n_agents]
            logit = logit.log_softmax(dim=-1)  # [batch_size, n_agents, n_nodes]
            if mode == "greedy":
                allocated_agent = agent_intention.argmax(dim=-1)  # [batch_size]
            elif mode == "sample":
                allocated_agent = Categorical(probs=agent_intention).sample()  # [batch_size]
            elif mode == "epsilon greedy":
                if random.random() < self.allocate_epsilon:
                    allocated_agent = Categorical(probs=agent_intention).sample()
                else:
                    allocated_agent = agent_intention.argmax(dim=-1)
            else:
                raise NotImplementedError
            allocated_node = logit.gather(1, allocated_agent.unsqueeze(-1).unsqueeze(-1).repeat(1, 1,
                                                                                                self.n_nodes)).argmax(
                dim=-1)  # [batch_size]

        elif order == "node first":
            node_intention = torch.softmax(logit.sum(dim=1), dim=-1)  # [batch_size, n_nodes]
            logit = logit.log_softmax(dim=-1)  # [batch_size, n_agents, n_nodes]
            if mode == "greedy":
                allocated_node = node_intention.argmax(dim=-1)  # [batch_size]
            elif mode == "sample":
                allocated_node = Categorical(probs=node_intention).sample()  # [batch_size]
            elif mode == "epsilon greedy":
                if random.random() < self.allocate_epsilon:
                    allocated_node = Categorical(probs=node_intention).sample()
                else:
                    allocated_node = node_intention.argmax(dim=-1)
            else:
                raise NotImplementedError
            allocated_agent = logit.gather(2, allocated_node.unsqueeze(-1).unsqueeze(-1).repeat(1, self.n_agents,
                                                                                                1)).argmax(
                dim=1)  # [batch_size]

        log_p = logit.gather(1, allocated_agent.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.n_nodes)).gather(2,
                                                                                                               allocated_node.unsqueeze(
                                                                                                                   -1).unsqueeze(
                                                                                                                   -1).repeat(
                                                                                                                   1, 1,
                                                                                                                   self.n_nodes)).squeeze(
            -1).squeeze(-1)  # [batch_size]
        return allocated_node, allocated_agent, log_p  # [batch_size] [batch_size] [batch_size]

    def set_freeze_encoder(self):
        self.freezeGraphEncoder = GraphEncoder(self.n_layers, self.n_embed, self.node_dim, self.n_head, self.n_nodes,
                                         self.normalization)
        self.freezeGraphEncoder.load_state_dict(self.GraphEncoder.state_dict())
        self.freezeGraphEncoder.to(self.device)