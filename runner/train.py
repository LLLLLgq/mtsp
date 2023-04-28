import math
import os
import sys
import time
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
    model.set_decoder_mode("sample")
    step = 0
    train_dataset = problem.make_dataset(args.graph_size, args.epoch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=32)
    for batch_id, batch in enumerate(tqdm(train_loader, disable=args.no_progress_bar)):
        batch = batch.to(device)
        cost, log_p = model.get_tsp_solution(batch, 'tsp')  # cost:(B) , log_p: (B)
        baseline = torch.zeros_like(cost)
        bs_loss = 0
        if args.baseline == 'critic':
            baseline = model.get_val()
            bs_loss = F.mse_loss(cost, baseline)
        reinforce_loss = ((cost.detach()-baseline.detach()) * log_p).mean()
        wandb.log({"reinforce_loss": reinforce_loss.item(),
                   "bs_loss": bs_loss.item(),
                   "cost": cost.mean().item(),
                   "baseline": baseline.mean().item()})
        loss = reinforce_loss + args.critic_coef * bs_loss
        optimizer.zero_grad()
        loss.backward()
        grad_norm, grad_norms_clipped = clip_grad_norms(optimizer.param_groups, 1.0)
        optimizer.step()
        if args.log_interval != 0 and step % args.log_interval == 0:
            print("grad_norm: {}".format(grad_norm))
            print("step: {}, reinforce_loss: {}".format(step, reinforce_loss.item()))
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
    model.set_decoder_mode("sample")
    step = 0
    train_dataset = problem.make_dataset(args.graph_size, args.epoch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=32)
    for batch_id, batch in enumerate(tqdm(train_loader, disable=args.no_progress_bar)):
        batch = batch.to(device)
        if args.mtsp_autoregressive:
            assignment, log_p = model.divide_nodes(batch) # assignment: (B, A, N), log_p: (B)
        else:
            assignment, log_p = model.classify_node(batch)
        print(assignment)
        sys.exit(0)
        cost, log_p = model.get_tsp_solution(batch, 'mtsp')  # cost:(B) , log_p: (B)
        reinforce_loss = ((cost.detach()) * log_p).mean()
        optimizer.zero_grad()
        reinforce_loss.backward()
        grad_norm, grad_norms_clipped = clip_grad_norms(optimizer.param_groups, 1.0)
        optimizer.step()
        if args.log_interval != 0 and step % args.log_interval == 0:
            print("grad_norm: {}".format(grad_norm))
            print("step: {}, reinforce_loss: {}".format(step, reinforce_loss.item()))
            # if not args.no_log:
            #     writer.add_scalar('reinforce_loss', reinforce_loss.item(), step)
            #     if args.use_wandb:
            #         wandb.log({"reinforce_loss": reinforce_loss.item()})
        step += 1

    finished_time = time.time()
    print("epoch time: {}".format(finished_time - start_time))



def evaluate(model, problem, args, device):
    with torch.no_grad():
        model.eval()
        model.set_decoder_mode("greedy")
        eval_dataset = problem.make_dataset(args.eval_graph_size, args.eval_size)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=32)
        const_sum = []
        for batch in eval_loader:
            batch = batch.to(device)
            cost, log_p = model.get_tsp_solution(batch, 'tsp')
            const_sum.append(cost.mean())
    torch.cuda.empty_cache()
    print("tsp cost over {} nodes: {}".format(args.eval_graph_size, torch.stack(const_sum).mean()))
    if (not args.no_log) and args.use_wandb:
        wandb.log({"tsp eval_cost over {} nodes".format(args.eval_graph_size): torch.stack(const_sum).mean()})
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
    #device = torch.device("cuda:0" if not args.no_cuda else "cpu")
    device = torch.device("cpu")
    # initialize model
    model = AttentionModel(args).to(device)
    params_sum = 0
    for name, p in model.named_parameters():
        params_sum += p.numel()
        print('  [*] {} : {}'.format(name, p.numel()))
    print('  [*] Number of parameters: {}'.format(params_sum))
    # initialize baseline

    # initialize optimizer
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args.lr_actor}])
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: args.lr_decay ** epoch)
    train_mtsp_epoch(model, problem, optimizer, args, device)
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

    for epoch in range(args.n_epoch):
        print("Epoch: {}, lr: {}, Step;{}".format(epoch,
                                                  lr_scheduler.get_last_lr(),
                                                  epoch * (args.epoch_size // args.batch_size)))
        train_epoch(model,
                    problem,
                    optimizer,
                    args,
                    writer,
                    device)
        lr_scheduler.step()
        if args.eval_epoch != 0 and epoch % args.eval_epoch == 0:
            evaluate(model, problem, args, device)
        if not args.no_save and epoch % args.save_epoch == 0:
            path = os.path.join(args.save_dir, str(args.problem) + str(args.graph_size), "epoch_{}.pt".format(epoch))
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        'args': args}, path)

    if writer is not None:
        writer.close()
    return 0


if __name__ == '__main__':
    main(sys.argv[1:])
