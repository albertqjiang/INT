from algos.eval import eval_agent
from legacy.data_generation.random_numerical_specified import specify_problem
from data_generation.utils import Dataset, valid_combo
from proof_system.all_axioms import operator_axioms

import argparse
import os
import json
import random
import itertools
import torch
from copy import deepcopy

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
ap.add_argument("-otts", "--online_train_test_split", required=False, type=float, default=1.)

ap.add_argument('--model_paths', required=True, type=str, nargs="+", help='path to saving directories')
ap.add_argument('--combos', required=False, type=int, default=100, help="number of combinations")
ap.add_argument('--problems', required=False, type=int, default=10, help="number of problems per combination")
ap.add_argument('--axioms', required=True, type=int, help="number of axioms for each combination")
ap.add_argument('--length', required=True, type=int, help="length of problems for each combination")
args = ap.parse_args()


def generate_online_data(train_combos, test_combos):
    train_steps = []
    train_first_steps = []
    val_steps = []
    val_first_steps = []
    for train_combo in train_combos:
        for _ in range(int(args.problems * args.online_train_test_split)):
            steps = specify_problem(train_combo, args.length)
            train_steps.extend(deepcopy(steps))
            train_first_steps.extend(deepcopy([steps[0]]))
        for _ in range(int(args.problems * (1 - args.online_train_test_split))):
            steps = specify_problem(train_combo, args.length)
            val_steps.extend(deepcopy(steps))
            val_first_steps.extend(deepcopy([steps[0]]))
    random.shuffle(train_steps)
    random.shuffle(train_first_steps)
    random.shuffle(val_steps)
    random.shuffle(val_first_steps)
    train_dataset = Dataset(train_steps)
    train_first_dataset = Dataset(train_first_steps)
    val_dataset = Dataset(val_steps)
    val_first_dataset = Dataset(val_first_steps)

    test_steps = []
    test_first_steps = []
    for test_combo in test_combos:
        for _ in range(int(args.problems * (1 - args.online_train_test_split))):
            steps = specify_problem(test_combo, args.length)
            test_steps.extend(deepcopy(steps))
            test_first_steps.extend(deepcopy([steps[0]]))
    random.shuffle(test_steps)
    random.shuffle(test_first_steps)
    test_dataset = Dataset(test_steps)
    test_first_dataset = Dataset(test_first_steps)
    return {
        "train": train_dataset,
        "train_first": train_first_dataset,
        "val": val_dataset,
        "val_first": val_first_dataset,
        "test": test_dataset,
        "test_first": test_first_dataset
    }


# Test every epoch
def test_rollout(model, test_dataset):
    model.eval()
    indices = list(range(len(test_dataset)))
    test_indices = random.choices(indices, k=args.test_points)
    test_starting_points = [test_dataset.trajectories[ind] for ind in test_indices]
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
        "time_limit": args.time_limit
    }
    success_rate, wrong_cases, success_cases = \
        eval_agent(model, env_config=env_config)
    model.train()
    return success_rate, wrong_cases, success_cases


def get_train_and_test_combos():
    split = args.online_train_test_split
    all_numerical_keys = list(operator_axioms.keys())
    random.shuffle(all_numerical_keys)
    all_numerical_axioms = {key: operator_axioms[key] for key in all_numerical_keys}
    all_combinations = itertools.combinations(all_numerical_axioms.keys(), r=args.axioms)
    valid_combinations = []
    for combination in all_combinations:
        if len(valid_combinations) >= args.combos:
            break
        if valid_combo(combination):
            valid_combinations.append(
                {axiom_name: all_numerical_axioms[axiom_name] for axiom_name in combination}
            )
    random.shuffle(valid_combinations)
    train_combinations = valid_combinations[:int(split * len(valid_combinations))]
    test_combinations = valid_combinations[int(split * len(valid_combinations)):]
    return train_combinations, test_combinations


train_combos, test_combos = get_train_and_test_combos()
assert len(test_combos) == 0

first_step_dataset = generate_online_data(train_combos, test_combos)["train_first"]

for model_path in args.model_paths:
    models = torch.load(os.path.join(model_path, "models", "checkpoints.pt"))
    model = None
    while model is None:
        model = models.pop()
    model.eval()
    first_step_success_rate, first_step_wrong_case, first_step_right_case = \
        test_rollout(model, first_step_dataset)
    json.dump(first_step_success_rate,
              open(os.path.join(model_path, "k{}l{}.json".format(args.axioms, args.length)), "w"))
