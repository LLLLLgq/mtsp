import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import ttest_rel
from tqdm import tqdm


class Baseline(object):

    def __init__(self):
        super(Baseline, self).__init__()

    def eval(self, x, cost):
        raise NotImplementedError

    def epoch_callback(self, model, epoch):
        pass


class NoBaseline(Baseline):

    def __init__(self):
        super(Baseline, self).__init__()

    def eval(self, x, cost):
        return 0, 0


class Critic(Baseline):

    def __init__(self, critic):
        super(Baseline, self).__init__()
        self.critic = critic

    def eval(self, x, cost):
        # x:graph_embedding (B, D)
        v = self.critic(x)
        return v, F.mse_loss(v, cost.detach())

    def get_learnable_params(self):
        return self.critic.parameters()

    def get_state_dict(self):
        return self.critic.state_dict()

    def load_state_dict(self, state_dict):
        self.critic.load_state_dict(state_dict)


class RolloutBaseline(Baseline):

    def __init__(self, model, problem, args):
        super(Baseline, self).__init__()
        self.problem = problem
        self.rollout_size = args.rollout_size
        self.graph_size = args.graph_size
        self.args = args
        self._update_model(model)

    def _update_model(self, model):
        self.model = copy.deepcopy(model)
        self.dataset = self.problem.make_dataset(self.graph_size, self.rollout_size)
        self.rollout_val = self.rollout(model)
        self.rollout_mean = self.rollout_val.mean()

    def rollout(self, model):
        model.set_decoder_mode("greedy")
        model.eval()
        rollout_val = []
        for bat in tqdm(DataLoader(self.dataset, batch_size=self.args.batch_size), disable=self.args.no_progress_bar):
            cost, _ = self.model(bat)
            rollout_val.append(cost.detach().cpu())
        return torch.cat(rollout_val, dim=0).numpy()

    def eval(self, x, cost):
        # x : node (B, N, D)
        with torch.no_grad():
            v = self.model(x)
        return v, 0

    def epoch_callback(self, model, epoch):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        :param model: The model to challenge the baseline by
        :param epoch: The current epoch
        """
        print("Evaluating candidate model on evaluation dataset")
        candidate_val = self.rollout(model)
        candidate_mean = candidate_val.mean()

        print("Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
            epoch, candidate_mean, epoch, self.rollout_val, candidate_mean - self.rollout_val))

        if candidate_mean - self.rollout_val < 0:
            # Calc p value
            t, p = ttest_rel(candidate_val, self.rollout_val)

            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < self.args.rollout_alpha:
                print('Update baseline')
                self._update_model(model)
