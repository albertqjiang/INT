import argparse
import json
import os
import random
import pickle
import time
from time import gmtime
from time import strftime

from data_generation.generate_problems import generate_multiple_problems


# Use different random seed to generate problems simultaneously
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel generator')
    parser.add_argument('--orders_path',
                        default="/scratch/hdd001/home/ajiang/data/INT/ordered_field")
    parser.add_argument('--dump_path', '-dp')
    parser.add_argument('-k', type=int)
    parser.add_argument('-l', type=int)
    parser.add_argument('-seed', type=int)
    parser.add_argument('-noisy', action='store_true')
    parser.add_argument('--noisy_p', type=float, default=0.5)
    parser.add_argument('--degree', type=int, default=0)
    parser.add_argument('--num_probs', type=int, default=1)
    parser.add_argument('--use_combos', action='store_true')
    parser.add_argument('--num_order_or_combo', type=int, default=None)
    args = parser.parse_args()

    random.seed(args.seed)

    if not os.path.isdir(args.dump_path):
        os.makedirs(args.dump_path)

    combos = json.load(open(os.path.join(args.orders_path, "combinations.json"), "r"))
    orders = json.load(open(os.path.join(args.orders_path, "orders.json"), "r"))
    NO_PER_BATCH = 100
    time0 = time.time()
    for j in range(int(args.num_probs/(NO_PER_BATCH + 1))+1):
        if args.use_combos:
            datasets, problems = generate_multiple_problems(num_axioms=args.k, length=args.l,
                                                            num_probs=NO_PER_BATCH, train_test="train",
                                                            combos=combos, degree=args.degree,
                                                            num_order_or_combo=args.num_order_or_combo,
                                                            noisy=args.noisy, p=args.noisy_p)
        else:
            datasets, problems = generate_multiple_problems(num_axioms=args.k, length=args.l,
                                                            num_probs=NO_PER_BATCH, train_test="train",
                                                            orders=orders, degree=args.degree,
                                                            num_order_or_combo=args.num_order_or_combo,
                                                            noisy=args.noisy, p=args.noisy_p)
        pickle.dump(problems,
                    open(os.path.join(args.dump_path, "problems_seed_{}_part{}.pkl".format(args.seed, j+1)), "wb"))

        taken_time = time.time() - time0
        estimated_remaining_time = taken_time / (j + 1) * (int(args.num_probs/(NO_PER_BATCH + 1))+1) - taken_time
        print("Generated {} problems, took {}".format((j+1)*NO_PER_BATCH, strftime("%H:%M:%S", gmtime(taken_time))))
        print("Estimated time to finish: {}".format(strftime("%H:%M:%S", gmtime(estimated_remaining_time))))
        print()
