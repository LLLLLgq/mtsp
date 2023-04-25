import torch
from torch.utils.data import Dataset
from .state_mtsp import StateMTSP


class MTSP(object):
    NAME = 'tsp'

    @staticmethod
    def get_cost(self, dataset, solution):
        # dataset (B,N,2)
        # solution (B,N_Agent,N)
        B, A, _ = solution.size()
        sequence = solution.clone()
        solution = solution.view(solution.size(0), -1)
        solution = solution.unique(dim=1)
        assert (
                torch.arange(solution.size(1), out=solution.data.new()).view(1, -1).expand_as(solution) ==
                solution.data.sort(1)[0]
        ).all(), "Invalid tour"
        # dataset (B,N,2) solution (B,N_Agent,N)
        max_dist = torch.zeros(B)
        for i in range(A):
            d = torch.gather(dataset, 1, solution[:,i,:].unsqueeze(-1).expand_as(dataset))
            max_dist = max(max_dist, (d[:, 1:] - d[:, :-1]).norm(2, 2).sum(1) + (d[:, 0] - d[:, -1]).norm(2, 1))
        return max_dist

    @staticmethod
    def make_dataset(size, num_samples):
        return MTSPDataset(size=size, num_samples=num_samples)

    @staticmethod
    def make_state(*args, **kwargs):
        # loc (B,N,2)
        return StateMTSP.initialize(*args, **kwargs)

    @staticmethod
    def get_dist(self, *args, **kwargs):
        return StateMTSP.get_dist(*args, **kwargs)


class MTSPDataset(Dataset):

    def __init__(self, filename=None, size=100, num_samples=1000):
        super(MTSPDataset, self).__init__()
        self.size = size
        self.num_samples = num_samples
        self.tsp = MTSP()
        self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for _ in range(num_samples)]
        self.filename = filename

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
