from typing import NamedTuple

import torch


class StateMTSP(NamedTuple):
    loc: torch.Tensor  # (B, N, 2)
    dist: torch.Tensor  # (B, N, N)

    first_action: torch.Tensor  # (B, A)
    last_action: torch.Tensor  # (B, A)
    visited: torch.Tensor  # (B, N)
    lengths: torch.Tensor  # (B, A)
    cur_coord: torch.Tensor  # (B, A, 2)
    step: int  # (1)
    n_agent: int # (A)

    def __getitem__(self, key):
        return self._replace(
            first_action=self.first_action[key],
            last_action=self.last_action[key],
            visited=self.visited[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key]
        )

    @staticmethod
    def initialize(loc, n_agent):
        B, N, _ = loc.size()
        device = loc.device
        A = n_agent
        last_action = torch.zeros(B, A, dtype=torch.long, device=device)
        visited = torch.zeros(B, N, dtype=torch.bool, device=device)
        visited[:, 0] = True
        return StateMTSP(
            loc=loc,
            dist=torch.norm(loc[:, None, :, :] - loc[:, :, None, :], p=2, dim=-1),
            first_action=last_action,
            last_action=last_action,
            visited=visited,
            lengths=torch.zeros(B, A, device=device),
            cur_coord=loc[:, 0, :].repeat(1, A, 1),
            step=1
        )

    def update(self, actions):
        raise NotImplementedError
        # action: (B,A)
        cur_coord = self.loc.gather(1, actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.loc.size(-1)))
        lengths = self.lengths + self.dist.gather(1, self.last_action.unsqueeze(-1).expand(-1, -1, self.dist.size(-1))) \
            .gather(2, actions.unsqueeze(-1).unsqueeze(-1)).squeeze(-1)
        visited = self.visited.clone()
        visited = visited.scatter_(1, actions.unsqueeze(-1), True)
        step = self.step + 1
        first_action = self.first_action
        last_action = actions.unsqueeze(-1)  # (B, 1)
        return self._replace(
            first_action=first_action,
            last_action=last_action,
            cur_coord=cur_coord,
            lengths=lengths,
            visited=visited,
            step=step
        )

    def get_cost(self, solution):
        # solution: (B, N)
        d = self.loc.gather(dim = 1, index = solution.unsqueeze(-1).expand(-1, -1, self.loc.size(-1))) # (B, N, 2)

        return torch.norm(d[:, 1:, :] - d[:, :-1, :], p=2, dim=-1).sum(dim=-1) + torch.norm(d[:, 0, :] - d[:, -1, :], p=2, dim=-1)

    def get_current_node(self):
        return self.last_action, self.cur_coord

    def get_mask(self):
        return self.visited

    def get_dist(self):
        return self.dist