import torch
from torch.utils.data import Dataset
from .state_tsp import StateTSP


class TSP(object):
    NAME = 'tsp'

    @staticmethod
    def get_cost(self, dataset, solution):
        assert (
                torch.arange(solution.size(1), out=solution.data.new()).view(1, -1).expand_as(solution) ==
                solution.data.sort(1)[0]
        ).all(), "Invalid tour"
        # dataset (B,N,2) solution (B,N)
        d = torch.gather(dataset, 1, solution.unsqueeze(-1).expand_as(dataset))

        return (d[:, 1:] - d[:, :-1]).norm(2, 2).sum(1) + (d[:, 0] - d[:, -1]).norm(2, 1)

    @staticmethod
    def make_dataset(size, num_samples):
        return TSPDataset(size=size, num_samples=num_samples)

    @staticmethod
    def make_state(*args, **kwargs):
        # loc (B,N,2)
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def get_dist(self, *args, **kwargs):
        return StateTSP.get_dist(*args, **kwargs)


class TSPDataset(Dataset):

    def __init__(self, filename=None, size=100, num_samples=1000):
        super(TSPDataset, self).__init__()
        self.size = size
        self.num_samples = num_samples
        self.tsp = TSP()
        self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for _ in range(num_samples)]
        self.filename = filename

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
