import itertools
import random
from pprint import pprint
from collections import deque
import numpy as np
from random import shuffle
import sys
import os

sys.path.insert(0, os.path.abspath('../..'))
from logic.logic import *
from logic.utils import NumericalEvaluator, standard_numerical_functions

x = Entity("input1")
y = Entity("input2")
z = Entity("input3")
one = Entity("1", is_constant=True)
zero = Entity("0", is_constant=True)
symbol_entities = [x, y]
numerical_entities = [one, zero]

del standard_numerical_functions["identity"]
del standard_numerical_functions["geometric_mean"]
numeric_collection = [nf for nf in standard_numerical_functions.values()]

# The maximum degree for generating entities
MAX_DEGREE = 4

entities = symbol_entities + numerical_entities
final_entities = list()
nodes_to_explore = deque([(symbol, 1) for symbol in symbol_entities])
while nodes_to_explore:
    left_node = nodes_to_explore.popleft()
    for nf in numeric_collection:
        for operands in itertools.combinations(nodes_to_explore, r=nf.input_no - 1):
            total_degree = left_node[1]
            for operand in operands:
                total_degree += operand[1]
            if total_degree < MAX_DEGREE:
                execution = nf.execute_nf([left_node[0]] + [operand[0] for operand in operands])
                final_entities.append(execution)
                nodes_to_explore.append((execution, total_degree + 1))
            elif total_degree == MAX_DEGREE:
                execution = nf.execute_nf([left_node[0]] + [operand[0] for operand in operands])
                final_entities.append(execution)

print(len(final_entities))
pprint([entity.to_string() for entity in random.choices(final_entities, k=10)])

ne = NumericalEvaluator()
equal_entity_pairs = list()
unequal_entity_pairs = list()

for entity0, entity1 in itertools.combinations(final_entities, r=2):
    if np.allclose(ne.evaluate(entity0.name, len(symbol_entities)), ne.evaluate(entity1.name, len(symbol_entities))):
        equal_entity_pairs.append((entity0, entity1, True))
    else:
        unequal_entity_pairs.append((entity0, entity1, False))

simple_proof = pickle.load(open("../data/simple_proof.p", "rb"))
for gt in simple_proof.ground_truth:
    if gt.logic_function.name == "Equivalent" and len(gt.entities) == 2:
        equal_entity_pairs.append((gt.entities[0], gt.entities[1], True))

shuffle(unequal_entity_pairs)
unequal_entity_pairs = unequal_entity_pairs[:len(equal_entity_pairs)]

entity_pairs = equal_entity_pairs + unequal_entity_pairs
shuffle(entity_pairs)
print([(entity0.name, entity1.name, target) for entity0, entity1, target in entity_pairs[:10]])
print(len(entity_pairs))
pickle.dump(entity_pairs, open("../data/balanced_entities.pkl", "wb"))
