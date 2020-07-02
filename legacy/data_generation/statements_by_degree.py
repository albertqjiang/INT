import pickle
import os
import sys
import random

sizes = [10, 20, 50, 100]
sys.path.insert(0, os.path.abspath('../..'))

simple_proof = pickle.load(open("../data/simple_proof.p", "rb"))
max_degree = 10
statements_by_degree = {i: list() for i in range(1, 1 + max_degree)}
for gt in simple_proof.ground_truth:
    if gt.degree <= max_degree:
        statements_by_degree[gt.degree].append(gt)

for key, value in statements_by_degree.items():
    for size in sizes:
        pickle.dump(random.choices(value, k=size),
                    open("../data/objectives_by_degree/simple_proof_degree_{}_{}_objectives.p".format(key, size), "wb"))
