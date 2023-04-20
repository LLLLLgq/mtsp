from typing import NamedTuple

import torch


class StateTSP(NamedTuple):
    loc: torch.Tensor  # (B, N, 2)
    dist: torch.Tensor  # (B, N, N)

    first_action: torch.Tensor  # (B, 1)
    last_action: torch.Tensor  # (B, 1)
    visited: torch.Tensor  # (B, N)
    lengths: torch.Tensor  # (B, 1)
    cur_coord: torch.Tensor  # (B, 2)
    step: torch.Tensor  # (1)

    def __getitem__(self, key):
        return self._replace(
            first_action=self.first_action[key],
            last_action=self.last_action[key],
            visited=self.visited[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key]
        )

    @staticmethod
    def initialize(loc):
        B, N, _ = loc.size()
        device = loc.device
        last_action = torch.zeros(B, 1, dtype=torch.long, device=device)
        visited = torch.zeros(B, N, dtype=torch.bool, device=device)
        visited[:, 0] = True
        return StateTSP(
            loc=loc,
            dist=torch.norm(loc[:, None, :, :] - loc[:, :, None, :], p=2, dim=-1),
            first_action=last_action,
            last_action=last_action,
            visited=visited,
            lengths=torch.zeros(B, 1, device=device),
            cur_coord=loc[:, 0, :],
            step=torch.zeros(1, dtype=torch.int64, device=device)
        )

    def update(self, actions):
        # action: (B)
        cur_coord = self.loc.gather(1, actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.loc.size(-1)))
        temp = self.dist.gather(1, self.last_action.unsqueeze(-1).expand(-1, -1, self.dist.size(-1))) \
            .gather(2, actions.unsqueeze(-1).unsqueeze(-1)).squeeze(-1)
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