import argparse
import parser


def get_config(args=None):
    parser = argparse.ArgumentParser(
        description="embracing the skill of singular solver"
    )
    # Data
    parser.add_argument('--problem', default='tsp', help="the problem to solve")
    parser.add_argument('--m_problem', default='mtsp', help="the problem to solve")
    parser.add_argument('--n_agents', default=5, type=int, help="the number of agents, only multi-agent for mtsp")
    parser.add_argument('--graph_size', default=100, type=int, help="the size of graph")
    parser.add_argument('--eval_graph_size', default=100, type=int, help="the size of evaluation graph")
    parser.add_argument('--batch_size', default=1000, type=int, help="the size of batch")
    parser.add_argument('--epoch_size', default=1000000, type=int, help="the size of epoch")
    parser.add_argument('--eval_size', default=20000, type=int, help="the size of evaluation")

    # Model
    parser.add_argument('--n_layers', default=3, type=int,
                        help="the number of attention layers specifically for attention model")
    parser.add_argument('--n_head', default=8, type=int, help="the number of heads for attention model")
    parser.add_argument('--n_embed', default=128, type=int, help="the dimension of embedding")
    parser.add_argument('--n_hidden', default=128, type=int, help="the dimension of hidden")
    parser.add_argument('--encoder_normalization', default='layer', help="the normalization of encoder")
    parser.add_argument('--tanh_clipping', default=10.0, type=float, help="the clipping value for tanh")
    parser.add_argument('--node_dim', default=2, type=int, help="the dimension of node")
    parser.add_argument('--mtsp_autoregressive', action='store_true', help="whether use autoregressive model for mtsp")
    parser.add_argument('--decoder_mode', default='sample', help="the mode of decoder, sample or greedy or ε-greedy")
    parser.add_argument('--dec_actor', action='store_true', help="whether use attention model as actor")
    parser.add_argument('--epsilon', default=0.1, type=float, help="the probability of random action")
    parser.add_argument('--epsilon_decay', default=0.999, type=float, help="the decay of epsilon")
    parser.add_argument('--allocate_mode', default='sample', help="the mode of allocate, sample or greedy")
    parser.add_argument('--allocate_order', default='equally', help="the order of allocate, equally / node first / "
                                                                    "edge first / agent first")
    parser.add_argument('--classify_mode', default='sample', help="the mode of gather, sample or greedy")

    # Save
    parser.add_argument('--no_save', action='store_true', help="disable saving")
    parser.add_argument('--save_epoch', default=10, type=int, help="the interval of saving, 0 for disable saving")
    parser.add_argument('--save_dir', default='save', help="the directory to save the model")

    # Load
    parser.add_argument('--load', action='store_true', help="enable load the model")
    parser.add_argument('--load_path', default=None, help="the path to load the model")

    # Training
    parser.add_argument('--n_epoch', default=50, type=int, help="the number of epochs")
    parser.add_argument('--n_mtsp_epoch', default=100, type=int, help="the number of epochs")
    parser.add_argument('--lr_actor', default=1e-4, type=float, help="the learning rate")
    parser.add_argument('--lr_decay', default=1.0, type=float, help="the learning rate decay")
    parser.add_argument('--lr_critic', default=1e-4, type=float, help="the learning rate")
    parser.add_argument('--ema_beta', default=0.8, type=float, help="the exponential moving average beta")
    parser.add_argument('--eval_epoch', default=1, type=int, help="the interval of evaluation")
    parser.add_argument('--log_interval', default=100, type=int, help="the interval of logging")
    parser.add_argument('--log_dir', default='log', help="the directory to save the log")
    parser.add_argument('--train_tsp', action='store_true', help="whether train tsp")
    parser.add_argument('--train_mtsp', action='store_true', help="whether train mtsp")

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