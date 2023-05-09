import math
import os
import sys
import time
from pathlib import Path

import torch
import wandb
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.getcwd(), '../..'))
from mtsp.utils.util import clip_grad_norms
from mtsp.utils.config import get_config
from mtsp.problem.tsp.problem_tsp import TSP
from mtsp.model.attention_model import AttentionModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_epoch(model, problem, optimizer, args, writer, device=torch.device("cpu")):
    start_time = time.time()
    model.train()
    model.set_tsp_decoder_mode("sample")
    step = 0
    train_dataset = problem.make_dataset(args.graph_size, args.epoch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=32)
    for batch_id, batch in enumerate(tqdm(train_loader, disable=args.no_progress_bar)):
        batch = batch.to(device)
        cost, log_p = model.get_tsp_solution(batch)  # cost:(B) , log_p: (B)
        baseline = torch.zeros_like(cost)
        bs_loss = 0
        if args.baseline == 'critic':
            baseline = model.get_val()
            bs_loss = F.mse_loss(cost, baseline)
        reinforce_loss = ((cost.detach() - baseline.detach()) * log_p).mean()
        if not args.no_log:
            wandb.log({"reinforce_loss": reinforce_loss.item(),
                       "bs_loss": bs_loss.item(),
                       "cost": cost.mean().item(),
                       "baseline": baseline.mean().item()})
        loss = reinforce_loss + args.critic_coef * bs_loss
        optimizer.zero_grad()
        loss.backward()
        grad_norm, grad_norms_clipped = clip_grad_norms(optimizer.param_groups, 1.0)
        optimizer.step()
        # if args.log_interval != 0 and step % args.log_interval == 0:
        #     print("step: {}, reinforce_loss: {}".format(step, reinforce_loss.item()))
        # if not args.no_log:
        #     writer.add_scalar('reinforce_loss', reinforce_loss.item(), step)
        #     if args.use_wandb:
        #         wandb.log({"reinforce_loss": reinforce_loss.item()})
        step += 1

    finished_time = time.time()
    print("epoch time: {}".format(finished_time - start_time))


def train_mtsp_epoch(model, problem, optimizer, args, device=torch.device("cpu")):
    start_time = time.time()
    model.train()
    model.set_tsp_decoder_mode("greedy")
    step = 0
    train_dataset = problem.make_dataset(args.graph_size, args.epoch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=32)
    for batch_id, batch in enumerate(tqdm(train_loader, disable=args.no_progress_bar)):
        batch = batch.to(device)
        if args.mtsp_autoregressive:
            assignment, log_p = model.divide_nodes(batch)  # assignment: (B, A, N), log_p: (B)
        else:
            assignment, log_p = model.classify_node(batch)  # assignment: (B, A, N), log_p: (B)
        mtsp_cost = torch.zeros(batch.size(0)).to(device)
        with torch.no_grad():
            tsp_cost, _ = model.get_tsp_solution(batch)
        for i in range(args.n_agents):
            new_batch = batch.gather(1, assignment[:, i, :].unsqueeze(-1).repeat(1, 1, batch.size(-1)))
            with torch.no_grad():
                cost, _ = model.get_tsp_solution(new_batch)  # cost:(B) , log_p: (B)
            mtsp_cost = torch.max(mtsp_cost, cost)
        reinforce_loss = ((mtsp_cost.detach()) * log_p).mean()
        optimizer.zero_grad()
        reinforce_loss.backward()
        grad_norm, grad_norms_clipped = clip_grad_norms(optimizer.param_groups, 1.0)
        optimizer.step()
        if not args.no_log:
            wandb.log({"mtsp_reinforce_loss": reinforce_loss.item(),
                       "mtsp_cost": mtsp_cost.mean().item(),
                       "tsp_cost": tsp_cost.mean().item()})
        step += 1

    finished_time = time.time()
    print("epoch time: {}".format(finished_time - start_time))


def evaluate(model, problem, args, device):
    with torch.no_grad():
        model.eval()
        model.set_tsp_decoder_mode("greedy")
        eval_dataset = problem.make_dataset(args.eval_graph_size, args.eval_size)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=32)
        const_sum = []
        mtsp_cost_sum = []
        for batch in eval_loader:
            batch = batch.to(device)
            # tsp
            cost, log_p = model.get_tsp_solution(batch)
            const_sum.append(cost.mean())
            # mtsp
            if args.mtsp_autoregressive:
                assignment, log_p = model.divide_nodes(batch)  # assignment: (B, A, N), log_p: (B)
            else:
                assignment, log_p = model.classify_node(batch)  # assignment: (B, A, N), log_p: (B)
            mtsp_cost = torch.zeros(batch.size(0)).to(device)
            for i in range(args.n_agents):
                new_batch = batch.gather(1, assignment[:, i, :].unsqueeze(-1).repeat(1, 1, batch.size(-1)))
                cost, _ = model.get_tsp_solution(new_batch)  # cost:(B) , log_p: (B)
                mtsp_cost = torch.max(mtsp_cost, cost)
            mtsp_cost_sum.append(mtsp_cost)
    torch.cuda.empty_cache()
    print("tsp cost over {} nodes: {}".format(args.eval_graph_size, torch.stack(const_sum).mean()))
    print("mtsp cost over {} nodes: {}".format(args.eval_graph_size, torch.stack(mtsp_cost_sum).mean()))
    if (not args.no_log) and args.use_wandb:
        wandb.log({"tsp eval_cost over {} nodes".format(args.eval_graph_size): torch.stack(const_sum).mean()})
        wandb.log({"mtsp eval_cost over {} nodes {} agents".format(args.eval_graph_size, args.n_agents): torch.stack(
            mtsp_cost_sum).mean()})
    return 0


def save():
    raise NotImplementedError


def main(args):
    args = get_config(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.problem == 'tsp':
        problem = TSP()
    else:
        problem = None
        raise NotImplementedError
    args.problem = problem
    device = torch.device("cuda:0" if not args.no_cuda else "cpu")

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / "graph{}".format(args.graph_size) / "agents{}".format(args.n_agents)

    exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                     str(folder.name).startswith('run')]
    if len(exst_run_nums) == 0:
        curr_run = 'run1'
    else:
        curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    # initialize model
    model = AttentionModel(args).to(device)
    params_sum = 0
    for name, p in model.named_parameters():
        params_sum += p.numel()
        # print('  [*] {} : {}'.format(name, p.numel()))
    print('  [*] Number of parameters: {}'.format(params_sum))
    # initialize baseline

    # initialize optimizer
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args.lr_actor}])
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: args.lr_decay ** epoch)
    # initialize logger
    writer = None
    if not args.no_log:
        if args.log_dir is not None:
            writer = SummaryWriter(args.log_dir)
        else:
            writer = SummaryWriter()
        if args.use_wandb:
            wandb.init(project='TSP' + str(args.graph_size), config=args)
            wandb.watch(model)

    for epoch in range(args.n_epoch + args.n_mtsp_epoch):
        print("Epoch: {}, lr: {}, Step;{}".format(epoch,
                                                  lr_scheduler.get_last_lr(),
                                                  epoch * (args.epoch_size // args.batch_size)))
        if epoch < args.n_epoch:
            train_epoch(model,
                        problem,
                        optimizer,
                        args,
                        writer,
                        device)
        else:
            if epoch == args.n_epoch:
                model.set_freeze_encoder()
            train_mtsp_epoch(model, problem, optimizer, args, device)
        lr_scheduler.step()
        if args.eval_epoch != 0 and epoch % args.eval_epoch == 0:
            evaluate(model, problem, args, device)
        if not args.no_save and (epoch % args.save_epoch == 0 or epoch == args.n_epoch - 1):
            print('saving model at epoch {} .................'.format(epoch))
            save_dir = run_dir / 'models'
            if not save_dir.exists():
                os.makedirs(str(save_dir))
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        'args': args}, str(save_dir) + '/epoch{}.pt'.format(epoch))

    if writer is not None:
        writer.close()
    return 0


if __name__ == '__main__':
    main(sys.argv[1:])
