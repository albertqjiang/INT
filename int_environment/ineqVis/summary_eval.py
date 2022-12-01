import os
import torch
import argparse
import algos.eval
import copy
import collections
import json
import utils
from legacy.data_generation.random_numerical_specified import load_online_combo_and_length

cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='eval')
parser.add_argument("-l", "--load-path", required=True, default=None, type=str,
                    help="what custom algos to use")
parser.add_argument("--combo-path", required=False, default=None, type=str,
                    help="what custom algos to use")
parser.add_argument("--log-dir", required=False, default=None, type=str,
                    help="what custom algos to use")
parser.add_argument("--eval_type", required=False, default="sl", type=str)
args = parser.parse_args()

if args.eval_type == "sl":
    model_path = os.path.join(args.load_path, "model_checkpoint.pt")
elif args.eval_type == "rl":
    model_path = os.path.join(args.load_path, "models", "checkpoints.pt")
else:
    raise NotImplementedError

if cuda:
    model = torch.load(model_path,
                       map_location=lambda storage, loc: storage)
    if args.eval_type == "rl":
        model = model[0]
    model.cuda()
else:
    model = torch.load(model_path,
                       map_location='cpu')
    if args.eval_type == "rl":
        model = model[0]

dirs = ["k=1_l=1", "k=1_l=3", "k=1_l=5", "k=2_l=3", "k=2_l=5", "k=2_l=7", "k=3_l=5", "k=3_l=7", "k=3_l=9", "k=4_l=7",
        "k=4_l=9"]
# dirs = ["k=2_l=3", "k=2_l=5", "k=3_l=5"]
# dirs = ["k=1_l=2"] #, "k=2_l=5", "k=3_l=5"]
log_dir = os.path.join(args.log_dir, args.load_path.split("/")[-1])
if os.path.exists(log_dir):
    print("algos exists!")
else:
    os.mkdir(log_dir)

eval_rewards = collections.defaultdict(dict)
all_trajectories = dict()
for prob in dirs:
    train_c_and_l, test_c_and_l = load_online_combo_and_length(args.combo_path, [prob])
    train_env_config = {
        "max_theorems": 25,
        "max_ground_truth": 50,
        "max_objectives": 1,
        "max_operands": 4,
        "max_edges": 200,
        "max_nodes": 50,
        "max_node_types": 40,
        "max_configuration": 10,
        "backplay": False,
        "mode": "eval",
        "online": True,
        "combos_and_lengths": train_c_and_l,
        "eval_dataset": None,
        "batch_eval": False,
        "verbo": True,
        "obs_mode": "geometric",
        "bag_of_words": False,
        "time_limit": 20,
        "num_online_evals": 100
    }
    test_env_config = copy.deepcopy(train_env_config)
    test_env_config["combos_and_lengths"] = test_c_and_l
    train_sr, train_wc, train_sc = algos.eval.test_rollout(model, [], env_config=train_env_config)
    test_sr, test_wc, test_sc = algos.eval.test_rollout(model, [], env_config=test_env_config)
    eval_rewards[prob]["train"] = train_sr
    eval_rewards[prob]["test"] = test_sr
    all_trajectories[prob] = collections.defaultdict(dict)
    all_trajectories[prob]["train"]["right"] = train_sc
    all_trajectories[prob]["train"]["wrong"] = train_wc
    all_trajectories[prob]["test"]["right"] = test_sc
    all_trajectories[prob]["test"]["wrong"] = test_wc
    json.dump(eval_rewards, open(os.path.join(log_dir, "eval_rewards.json"), "w"))
    json.dump(all_trajectories, open(os.path.join(log_dir, "all_trajectories.json"), "w"))

    train_sc_str = utils.trajs_to_str(train_sc)
    train_wc_str = utils.trajs_to_str(train_wc)
    test_sc_str = utils.trajs_to_str(test_sc)
    test_wc_str = utils.trajs_to_str(test_wc)

    json.dump(train_sc_str, open(os.path.join(args.load_path, "{}_train_success_cases.json".format(prob)), "w"))
    json.dump(train_wc_str, open(os.path.join(args.load_path, "{}_train_wrong_cases.json".format(prob)), "w"))
    json.dump(test_sc_str, open(os.path.join(args.load_path, "{}_test_success_cases.json".format(prob)), "w"))
    json.dump(test_wc_str, open(os.path.join(args.load_path, "{}_test_wrong_cases.json".format(prob)), "w"))
