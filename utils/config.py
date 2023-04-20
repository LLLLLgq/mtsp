import argparse
import parser


def get_config(args=None):
    parser = argparse.ArgumentParser(
        description="embracing the skill of singular solver"
    )
    # Data
    parser.add_argument('--problem', default='tsp', help="the problem to solve")
    parser.add_argument('--graph_size', default=50, type=int, help="the size of graph")
    parser.add_argument('--eval_graph_size', default=50, type=int, help="the size of evaluation graph")
    parser.add_argument('--batch_size', default=64, type=int, help="the size of batch")
    parser.add_argument('--epoch_size', default=1000000, type=int, help="the size of epoch")
    parser.add_argument('--eval_size', default=10000, type=int, help="the size of evaluation")

    # Model
    parser.add_argument('--n_layers', default=3, type=int,
                        help="the number of attention layers specifically for attention model")
    parser.add_argument('--n_head', default=8, type=int, help="the number of heads for attention model")
    parser.add_argument('--n_embed', default=128, type=int, help="the dimension of embedding")
    parser.add_argument('--n_hidden', default=128, type=int, help="the dimension of hidden")
    parser.add_argument('--encoder_normalization', default='layer', help="the normalization of encoder")
    parser.add_argument('--tanh_clipping', default=10.0, type=float, help="the clipping value for tanh")
    parser.add_argument('--node_dim', default=2, type=int, help="the dimension of node")
    parser.add_argument('--decoder_mode', default='sample', help="the mode of decoder, sample or greedy")

    # Save
    parser.add_argument('--no_save', action='store_true', help="disable saving")
    parser.add_argument('--save_interval', default=100, type=int, help="the interval of saving, 0 for disable saving")
    parser.add_argument('--save_dir', default='save', help="the directory to save the model")

    # Training
    parser.add_argument('--lr_actor', default=1e-4, type=float, help="the learning rate")
    parser.add_argument('--lr_decay', default=1.0, type=float, help="the learning rate decay")
    parser.add_argument('--lr_critic', default=1e-4, type=float, help="the learning rate")
    parser.add_argument('--n_epoch', default=100, type=int, help="the number of epochs")
    parser.add_argument('--ema_beta', default=0.8, type=float, help="the exponential moving average beta")
    parser.add_argument('--eval_epoch', default=1, type=int, help="the interval of evaluation")
    parser.add_argument('--log_interval', default=100, type=int, help="the interval of logging")
    parser.add_argument('--load_dir', default='save', help="the directory to load the model")
    parser.add_argument('--log_dir', default='log', help="the directory to save the log")

    # Baseline
    parser.add_argument('--baseline', default='critic', help="the baseline to use")
    parser.add_argument('--critic_coef', default=0.01, type=float, help="the coefficient of critic")
    parser.add_argument('--rollout_size', default=5000, type=int, help="the size of rollout")
    parser.add_argument('--rollout_alpha',type=float, default=0.05,
                        help='Significance in the t-test for updating rollout baseline')

    # Misc
    parser.add_argument('--seed', default=998244353, type=int, help="the random seed")
    parser.add_argument('--no_cuda', action='store_true', help="disable cuda")
    parser.add_argument('--use_wandb', action='store_false', help="use wandb")
    parser.add_argument('--use_lr_scheduler', action='store_false', help="use learning rate scheduler")
    parser.add_argument('--no_log', action='store_true', help="disable logging")
    parser.add_argument('--no_load', action='store_true', help="disable loading")
    parser.add_argument('--no_eval', action='store_true', help="disable evaluation")
    parser.add_argument('--no_train', action='store_true', help="disable training")
    parser.add_argument('--no_tensorboard', action='store_true', help="disable tensorboard")
    parser.add_argument('--no_progress_bar', action='store_true', help="disable progress bar")
    config = parser.parse_args(args)
    return config

if __name__ == '__main__':
    get_config()