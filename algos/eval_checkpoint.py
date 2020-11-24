from algos.eval import eval_agent
from data_generation.generate_problems import generate_multiple_problems
from data_generation.utils import Dataset

import argparse
import os
import json
import random
import torch
import numpy as np
import pickle

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

ap = argparse.ArgumentParser()
# Environment setting
ap.add_argument("--max_theorems", required=False, type=int, default=25,
                help="maximum number of theorems")
ap.add_argument("--max_ground_truth", required=False, type=int, default=50,
                help="maximum number of ground truth")
ap.add_argument("--max_objectives", required=False, type=int, default=1,
                help="maximum number of objectives")
ap.add_argument("--max_operands", required=False, type=int, default=4,
                help="maximum number of operands")
ap.add_argument("--max_edges", required=False, type=int, default=200,
                help="maximum number of edges in graphs")
ap.add_argument("--max_nodes", required=False, type=int, default=50,
                help="maximum number of nodes in graphs")
ap.add_argument("--max_node_types", required=False, type=int, default=40,
                help="maximum number of node types in graphs")
ap.add_argument("--max_configuration", required=False, type=int, default=10,
                help="maximum number of configurations in graphs")

# Neural net related
ap.add_argument("-bag", "--bag_of_words", required=False, type=bool, default=False,
                help="whether to use bag of words model")
ap.add_argument("-om", "--obs_mode", required=False, type=str, default="geometric",
                help="which mode of observation to use")

# Eval related
ap.add_argument("-ti", "--time_limit", required=False, type=int, default=20,
                help="time limit for evaluation")
ap.add_argument("-tp", "--test_points", required=False, type=int, default=1000,
                help="how many points to test on")
ap.add_argument("-du", "--dump", required=True, type=str, default=None,
                help="dump path")
ap.add_argument("-tts", "--train_test_split", required=False, type=float, default=0.9)
ap.add_argument("--train_test", required=False, type=str, default="train")
ap.add_argument("--seed", required=False, type=int, default=10,
                help="seed")

ap.add_argument('--model_directory', required=True, type=str, help='path to the parent algos of the checkpoints')
ap.add_argument('--earliest_timestamp', required=True, type=str, help='earliest timestamp to evaluate')
ap.add_argument('--latest_timestamp', required=True, type=str, help='latest timestamp to evaluate')
ap.add_argument("-trs", "--train_sets", required=True, type=str, nargs="+", default=["k=3_l=5"])
ap.add_argument('--combo_path', required=True, type=str)
ap.add_argument('--num_probs', required=False, type=int, default=100, help="number of problems per combination")
ap.add_argument('--num_axioms', required=True, type=int, help="number of axioms for each combination")
ap.add_argument('--num_length', required=True, type=int, help="length of problems for each combination")
ap.add_argument('--num_order_or_combo', required=False, type=int, default=-1,
                help="number of axiom order or combo used in train")
ap.add_argument('--last_num_order_or_combo', required=False, type=int, default=-1,
                help="number of last axiom order or combo used in eval")
ap.add_argument("-oog", "--online_order_generation", action='store_true',
                help="whether to use the axiom combinations to generate orders on the fly")
ap.add_argument('--degree', required=False, type=int, default=0, help="degree of initial conditions")
args = ap.parse_args()


# Test every epoch
def test_rollout(model, test_dataset):
    model.eval()
    test_starting_points = test_dataset.trajectories
    env_config = {
        "max_theorems": args.max_theorems,
        "max_ground_truth": args.max_ground_truth,
        "max_objectives": args.max_objectives,
        "max_operands": args.max_operands,
        "max_edges": args.max_edges,
        "max_nodes": args.max_nodes,
        "max_node_types": args.max_node_types,
        "max_configuration": args.max_configuration,
        "backplay": False,
        "mode": "eval",
        "eval_dataset": test_starting_points,
        "batch_eval": False,
        "verbo": True,
        "online": False,
        "obs_mode": args.obs_mode,
        "bag_of_words": args.bag_of_words,
        "time_limit": args.time_limit,
        "degree": args.degree
    }
    success_rate, wrong_cases, success_cases, avg_num_steps = \
        eval_agent(model, env_config=env_config)
    return success_rate, wrong_cases, success_cases, avg_num_steps


if __name__ == "__main__":
    torch.manual_seed(args.seed)
    if use_gpu:
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(0)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.num_order_or_combo < 0:
        args.num_order_or_combo = None

    if args.online_order_generation:
        axiom_combo_info_path = os.path.join(args.combo_path, "combinations.json")
    else:
        axiom_combo_info_path = os.path.join(args.combo_path, "orders.json")

    kl_dict = json.load(open(axiom_combo_info_path, "r"))
    if args.last_num_order_or_combo > 0:
        assert args.train_test == "train"
        trial_key = "k{}l{}".format(args.num_axioms, args.num_length)
        if trial_key in kl_dict:
            pass
        else:
            trial_key = "k{}".format(args.num_axioms)
            assert trial_key in kl_dict
        kl_dict[trial_key] = kl_dict[trial_key][-args.last_num_order_or_combo:]
        args.num_order_or_combo = len(kl_dict[trial_key])
    else:
        args.last_num_order_or_combo = None

    axiom_set = args.combo_path.split("/")[-1].split("_")[0]

    data_path = os.path.join(args.dump, "{}_eval_dataset_prob{}_k{}l{}_oog{}_{}_nooc{}_degree{}_lnoc{}.pkl".format(
        axiom_set, args.num_probs, args.num_axioms, args.num_length,
        args.online_order_generation, args.train_test, args.num_order_or_combo, args.degree,
        args.last_num_order_or_combo)
                             )
    if not os.path.isdir(args.dump):
        os.makedirs(args.dump)
    if os.path.isfile(data_path):
        eval_dataset = pickle.load(open(data_path, "rb"))
    else:
        if args.online_order_generation:
            keyword_arguments = {"combos": kl_dict}
        else:
            keyword_arguments = {"orders": kl_dict}
        _, separate_problems = generate_multiple_problems(num_axioms=args.num_axioms, length=args.num_length,
                                                          num_probs=args.num_probs,
                                                          train_test=args.train_test, backwards=True,
                                                          degree=args.degree,
                                                          online_order_generation=args.online_order_generation,
                                                          **keyword_arguments)
        extracted_first_steps = [steps[0] for steps in separate_problems]
        eval_dataset = Dataset(extracted_first_steps)
        pickle.dump(eval_dataset, open(data_path, "wb"))

    for timestamp in os.listdir(args.model_directory):
        if args.earliest_timestamp <= timestamp <= args.latest_timestamp:
            model_path = os.path.join(args.model_directory, timestamp)
            config_path = os.path.join(model_path, "env_config.json")
            if os.path.isfile(config_path):
                env_config = json.load(open(config_path, "r"))
                trained_kl = env_config["train_sets"][0]
                trained_k = int(trained_kl.split("_")[0][-1])
                trained_l = int(trained_kl[-1])
                if "online_order_generation" in env_config:
                    online_order_generation = env_config["online_order_generation"]
                else:
                    online_order_generation = False

                if "num_order_or_combo" in env_config:
                    num_order_or_combo = env_config["num_order_or_combo"]
                else:
                    num_order_or_combo = None

                degree = env_config["degree"]

                if axiom_set in env_config["combo_path"] and env_config["train_sets"][0] in args.train_sets:
                    checkpoint_path = os.path.join(model_path, "model_checkpoint.pt")
                    state_dict_path = os.path.join(model_path, "best_epoch.pth")
                    if os.path.isfile(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
                        model = torch.load(checkpoint_path, map_location="cuda")

                        if os.path.isfile(state_dict_path) and os.path.getsize(state_dict_path) > 0:
                            state_dict = torch.load(state_dict_path)
                            model.load_state_dict(state_dict['model'])
                        model.eval()

                        prefix = "train_k{}l{}_{}_oog{}_nooc{}_degree_{}_eval_k{}l{}_oog{}_{}_axiom_order_nooc{}_{}probs_degree{}_lnoc{}_{}".format(
                            trained_k, trained_l, axiom_set, online_order_generation, num_order_or_combo, degree,
                            args.num_axioms, args.num_length, args.online_order_generation, args.train_test,
                            args.num_order_or_combo, args.num_probs, args.degree, args.last_num_order_or_combo,
                            timestamp
                        )
                        sr_path = os.path.join(args.dump, "{}success_rate.json".format(prefix))
                        ans_path = os.path.join(args.dump, "{}avg_num_steps.json".format(prefix))

                        first_step_success_rate, _, _, avg_num_steps = test_rollout(model, eval_dataset)

                        json.dump(first_step_success_rate, open(sr_path, "w"))
                        json.dump(avg_num_steps, open(ans_path, "w"))
