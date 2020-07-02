import pickle
import json
import sys
import os

sys.path.insert(0, os.path.abspath('../..'))
from pprint import pprint
from legacy.helper_functions.utils import load_config, HingeLoss

import torch

# Parameters
ENTITY_DATA_PATH = "../data/balanced_entities.pkl"
config = load_config()["recurrent"]
HINGE_MIN = config["HINGE_MIN"]
HINGE_MAX = float(config["HINGE_MAX"])
MODEL_PATH = config["MODEL_PATH"]
DUMP_PATH = config["DUMP_PATH"]
TRAINING_TEST_SPLIT = 0.9

# Get dataset
entity_pair_list = pickle.load(open(ENTITY_DATA_PATH, "rb"))
dataset_size = len(entity_pair_list)
training_entity_pairs = entity_pair_list[:int(TRAINING_TEST_SPLIT * dataset_size)]
test_entity_pairs = entity_pair_list[int(TRAINING_TEST_SPLIT * dataset_size):]

model = torch.load(MODEL_PATH)["model"]
criterion = HingeLoss(min_val=HINGE_MIN, max_val=HINGE_MAX)
sorting_list = list()
list_length = len(test_entity_pairs)

for pair in test_entity_pairs[:list_length]:
    output1 = model.encode_entity(pair[0])
    output2 = model.encode_entity(pair[1])
    output = output1 - output2
    norm = torch.norm(output, p=1, dim=-1)
    sorting_list.append((norm.item(), pair[-1], pair[0].name, pair[1].name))

sorting_list.sort()
pprint(sorting_list[:int(list_length / 2)])
pprint(sorting_list[int(list_length / 2):])
json.dump(sorting_list[:int(list_length / 2)], open(DUMP_PATH + "first_half.json", "w"))
json.dump(sorting_list[int(list_length / 2):], open(DUMP_PATH + "second_half.json", "w"))

print(sum([1 for tup in sorting_list[:int(list_length / 2)] if tup[-1]]) / int(list_length / 2))
print(sum([1 for tup in sorting_list[int(list_length / 2):] if not tup[-1]]) / int(list_length / 2))
