from legacy.data_generation.exhaust_comp_graph import *
from legacy.data_generation.evaluate_graph import evaluate

import numpy as np
from time import time

pgs = exhaust_graph(degree=4)

num_tuple = np.load("../data/2inputs1tuples.npy")
print(num_tuple.shape)
timer = time()

for i in range(len(pgs)):
    pgs[i] = (pgs[i], evaluate(pgs[i], num_tuple=num_tuple))
    if i % 100 == 0:
        print("Handled{} graphs, took {}s.".format(i, time() - timer))

np_results = np.transpose(np.concatenate([pg[1] for pg in pgs], axis=1))
np.save("../data/2input4degreeevaluation", np_results)

unique_results = []

for i in range(len(np_results)):
    print(i)
    is_unique = True
    for j in range(len(unique_results)):
        if np.allclose(np_results[i], unique_results[j]):
            is_unique = False
            break
    if is_unique:
        unique_results.append(np_results[i])

print(len(unique_results))

print(np.unique(np_results, axis=0))
