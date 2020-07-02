from logic.logic import *

import itertools
import random
from collections import deque

r = LogicFunction("Real", 1)
equal = LogicFunction("Equals", 2)
nonNeg = LogicFunction("NonNegative", 1)
biggerOrEqual = LogicFunction("BiggerOrEqual", 2)
logic_collection = [r, equal, nonNeg, biggerOrEqual]

add = NumericalFunction("add", 2)
sub = NumericalFunction("sub", 2)
mul = NumericalFunction("mul", 2)
sqr = NumericalFunction("sqr", 1)
inv = NumericalFunction("inv", 1)
numeric_collection = [add, sub, mul, sqr, inv]

x = Entity("input1")
y = Entity("input2")

MAX_DEGREE = 5

with open("../data/logic_statement.txt", "w") as outh:
    entities = [x, y]
    final_entities = list()
    nodes_to_explore = deque([(x, 1), (y, 1)])
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

    print([e.to_string() for e in random.choices(final_entities, k=10)])

    for lf in logic_collection:
        input_no = lf.input_no
        entity_combinations = itertools.combinations(final_entities, input_no)
        for pair in entity_combinations:
            statement = lf.execute_lf(pair)
            outh.write(statement + "\t" + statement + "\n")

    # pprint(random.choices(statements, k=1))
