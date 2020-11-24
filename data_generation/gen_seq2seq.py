import argparse
import json
import os
import random
import pickle

from data_generation.generate_problems import generate_multiple_problems
from proof_system.graph_seq_conversion import Parser


random.seed(213)
proof_parser = Parser()


def convert_proof_to_seq2seq(steps):
    sources, targets = list(), list()
    for i, step in enumerate(steps):
        source, target = proof_parser.parse_proof_step_to_seq(step)
        sources.append(source)
        targets.append(target)
    return sources, targets


def generate_multiple_seq2seq(multiple_problems, all_sources_to_targets=None):
    if not all_sources_to_targets:
        all_sources_to_targets = dict()

    for problem in multiple_problems:
        sources, targets = convert_proof_to_seq2seq(problem)
        for source, target in zip(sources, targets):
            if source in all_sources_to_targets:
                continue
            all_sources_to_targets[source] = target
    return all_sources_to_targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mode generator')
    parser.add_argument('--orders_path',
                        default="/scratch/hdd001/home/ajiang/data/INT/ordered_field")
    parser.add_argument('--dump_path', '-dp', default="/scratch/hdd001/home/ajiang/data/INT/seq_exec")
    parser.add_argument('-k', type=int)
    parser.add_argument('-l', type=int)
    parser.add_argument('--degree', type=int, default=0)
    parser.add_argument('--num_probs', type=int, default=1)
    parser.add_argument('--use_combos', action='store_true')
    parser.add_argument('--num_order_or_combo', type=int, default=None)
    args = parser.parse_args()

    if not os.path.isdir(args.dump_path):
        os.makedirs(args.dump_path)

    combos = json.load(open(os.path.join(args.orders_path, "combinations.json"), "r"))
    orders = json.load(open(os.path.join(args.orders_path, "orders.json"), "r"))
    if args.num_probs > 10000:
        sources_to_targets = None
        for j in range(int(args.num_probs/1000)):
            if args.use_combos:
                datasets, problems = generate_multiple_problems(num_axioms=args.k, length=args.l,
                                                                num_probs=1000, train_test="train",
                                                                combos=combos, degree=args.degree,
                                                                num_order_or_combo=args.num_order_or_combo)
            else:
                datasets, problems = generate_multiple_problems(num_axioms=args.k, length=args.l,
                                                                num_probs=1000, train_test="train",
                                                                orders=orders, degree=args.degree,
                                                                num_order_or_combo=args.num_order_or_combo)
            sources_to_targets = generate_multiple_seq2seq(multiple_problems=problems,
                                                           all_sources_to_targets=sources_to_targets)
            pickle.dump(problems, open(os.path.join(args.dump_path, "problems_part{}.pkl".format(j+1)), "wb"))
    else:
        datasets, problems = generate_multiple_problems(num_axioms=args.k, length=args.l,
                                                        num_probs=args.num_probs, train_test="train",
                                                        orders=orders, degree=args.degree)
        sources_to_targets = generate_multiple_seq2seq(multiple_problems=problems)
        pickle.dump(problems, open(os.path.join(args.dump_path, "problems.pkl"), "wb"))

    randomised_keys = list(sources_to_targets.keys())
    random.shuffle(randomised_keys)

    if not os.path.isdir(args.dump_path):
        os.makedirs(args.dump_path)
    with open(os.path.join(args.dump_path, "all.src"), "w") as src_out:
        with open(os.path.join(args.dump_path, "all.tgt"), "w") as tgt_out:
            for key in randomised_keys:
                src_out.write(key)
                src_out.write("\n")
                tgt_out.write(sources_to_targets[key])
                tgt_out.write("\n")
