import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='ThmProving')
    # datasets
    parser.add_argument("-d", "--path_to_data", required=False, type=str,
                        help="what data algos to use")
    parser.add_argument("-trs", "--train_sets", required=True, type=str, nargs="+", default=["k=3_l=5"])
    parser.add_argument("-tes", "--test_sets", required=True, type=str, nargs="+", default=["k=3_l=5"])
    parser.add_argument("-o", "--obs_mode", required=False, type=str, default="geometric",
                        help="which mode of observation to use")
    parser.add_argument("-np", '--num_probs', required=True, type=int, default=1000,
                        help="number of problems per combination")
    parser.add_argument("-es", "--evaluation_size", required=False, type=int, default=256,
                        help="how many points to validate on")
    parser.add_argument("-nvp", "--num_val_probs", required=False, type=int, default=100,
                        help="how many points to validate on")
    parser.add_argument("-ntp", "--num_test_probs", required=False, type=int, default=100,
                        help="how many points to test on")
    parser.add_argument("-tg", "--transform_gt", action='store_true',
                        help="whether to use transform_gt")
    parser.add_argument("--online", required=False, action='store_true', default=True)
    parser.add_argument('-cp', "--combo_path", required=True, type=str,
                        default="/scratch/hdd001/home/ajiang/data/benchmark/random_specified")
    parser.add_argument("-oog", "--online_order_generation", action='store_true',
                        help="whether to use the axiom combinations to generate orders on the fly")
    parser.add_argument("-nooc", "--num_order_or_combo", required=False, type=int, default=-1,
                        help="number of orders or combos to use")

    # training settings
    parser.add_argument("--cuda", action='store_true',
                        help="how many total updates")
    parser.add_argument("-du", "--dump", required=True, type=str,
                        help="what dumping algos to use")
    parser.add_argument("-rd", "--resume_dir", required=False, default="", type=str,
                        help="what custom algos to use")
    parser.add_argument("-pr", "--pretrain_dir", required=False, default="", type=str,
                        help="what algos to load pretrain model")
    parser.add_argument("-fp", "--fix_policy", action='store_true',
                        help="whether to fix policy and train baseline for the first part of training")
    parser.add_argument("-epr", "--epoch_per_record", required=False, type=int, default=1,
                        help="how many epochs per record")
    parser.add_argument("-epcr", "--epoch_per_case_record", required=False, type=int, default=10,
                        help="how many epochs per record for right and wrong cases")
    parser.add_argument("-epod", "--epochs_per_online_dataset", required=False, type=int, default=10)
    parser.add_argument("-e", "--epoch", required=False, type=int, default=100000,
                        help="how many epochs")
    parser.add_argument("-u", "--updates", required=False, type=int, default=200000,
                        help="how many total updates")
    parser.add_argument("--time_limit", required=False, type=int, default=15)
    parser.add_argument("--seed", required=False, type=int, default=0)

    # optimization hps
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate (default: 7e-4)')
    parser.add_argument("-bs", "--batch_size", required=False, type=int, default=32,
                        help="what batch size to use")
    parser.add_argument("-le", "--lemma_cost", required=False, type=float, default=1.0,
                        help="lemma cost")
    parser.add_argument("-ent", "--entity_cost", required=False, type=float, default=1.0,
                        help="entity cost")
    parser.add_argument("-dr", "--dropout_rate", required=False, type=float, default=0.0,
                        help="dropout rate")
    parser.add_argument("-gdr", "--gat_dropout_rate", required=False, type=float, default=0.0,
                        help="dropout rate in gat")

    # neural architecture hps
    parser.add_argument("-hi", "--hidden", required=False, default=6, type=int,
                        help="how many hidden layers of nn")
    parser.add_argument("-hd", "--hidden_dim", required=False, type=int, default=512,
                        help="what state dimension to use")
    parser.add_argument("-gt", "--gnn_type", required=False, type=str, default="GIN",
                        help="what type of GNN to use")
    parser.add_argument("-atten", "--atten_type", type=int, required=False, default=0,
                        help="attention type")
    parser.add_argument("-ah", "--attention_heads", type=int, required=False, default=8,
                        help="attention heads")
    parser.add_argument("-n", "--norm", required=False, type=str, default=None,
                        help="what norm to use")
    # TODO: change this boolean arugment to be false
    parser.add_argument("-cgo", "--combined_gt_obj", action='store_false',
                        help="whether to use a combined gt and obj encoder")
    parser.add_argument("-bag", "--bag_of_words", action='store_true',
                        help="whether to use bag of words model")

    # Environment setting
    parser.add_argument("-m", "--mode", required=False, default="solve", type=str,
                        help="whether to discover or to solve")
    parser.add_argument("--verbo", required=False, action='store_true',
                        help="whether to use verbo")
    parser.add_argument("--degree", required=False, type=int, default=0,
                        help="the degree of the starting entities")

    # RL specific setting

    parser.add_argument('--eval_interval', type=int, default=None,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--algo', default='a2c', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--gail', action='store_true', default=False,
                        help='do imitation learning with gail')
    parser.add_argument('--gail-experts-dir', default='./gail_experts',
                        help='algos that contains expert demonstrations for gail')
    parser.add_argument('--gail-batch-size', type=int, default=128,
                        help='gail batch size (default: 128)')
    parser.add_argument('--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument("--num_processes", required=False, type=int, default=5,
                        help="the number of parallel processes")
    parser.add_argument('--num_steps', type=int, default=4,
                        help='number of forward steps in A2C (default: 4)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one l per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one s  per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='eval interval, one e  per n updates (default: None)')
    parser.add_argument('--num-env-steps', type=int, default=10e6,
                        help='number of environment steps to train (default: 10e6)')
    parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--saving_dir', default='/tmp/gym/',
                        help='algos to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='algos to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--use-proper-time-limits', action='store_true', default=False,
                        help='compute returns taking into account time limits')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False,
                        help='use a linear schedule on the learning rate')

    args = parser.parse_args()

    return args
