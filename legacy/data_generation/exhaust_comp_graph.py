from collections import deque
from copy import deepcopy
import itertools
from time import time
import pickle
import os.path

from legacy.pseq.graph import *

two_input_ops = [Addition, Multiplication]
one_input_ops = [SquareRoot, Reciprocal]


# TODO: Solve the problem of not enough memory
def exhaust_graph(input_num=2, degree=2, graphs_needed=1000, node_array_max_length=10000):
    """

    :param input_num: The number of inputs used for the graph
    :param degree: How many operation are used
    :return: A list of generated graphs with input_num inputs and less than degree operations with BFS
    """

    enough_graphs = False

    # If the graphs have been generated before, just take the result instead of generating them again
    if os.path.isfile("../data/{}input{}degree.pkl".format(input_num, degree)):
        with open("../data/{}input{}degree.pkl".format(input_num, degree), 'rb') as f:
            return pickle.load(f)

    # Generate inputs with names input1, 2, 3...
    inputs = [IOunit(Symbol(name="input{}".format(i), expression="input{}".format(i))) for i in range(1, input_num + 1)]
    possible_graphs = []
    nodes_to_visit = deque()
    nodes_to_visit.append((Graph(name="initialG", inputs=inputs), 0))
    print(possible_graphs, nodes_to_visit)

    timer = time()

    while nodes_to_visit:
        node = nodes_to_visit.popleft()
        graph = node[0]
        leaves = graph.get_num_of_leaves()

        if leaves > degree + 1 - node[1]:
            continue
        elif leaves == 1:
            if len(possible_graphs) >= graphs_needed:
                break
            possible_graphs.append(deepcopy(graph))
            if len(possible_graphs) % 100 == 0:
                print(time() - timer, "degree{}".format(node[1]))
                print("There are {} graphs and {} nodes to visit.".format(len(possible_graphs), len(nodes_to_visit)))

        if node[1] < degree:
            if len(nodes_to_visit) >= node_array_max_length:
                nodes_to_visit.append(node)
            else:
                for oio in one_input_ops:
                    for i in range(len(graph.get_iounits())):
                        graph_copy = deepcopy(graph)
                        io = graph_copy.get_iounits()[i]
                        operation = oio([io])
                        graph_copy.add_operation([operation])
                        nodes_to_visit.append((deepcopy(graph_copy), node[1] + 1))

                for tio in two_input_ops:
                    combinations = list(itertools.combinations(range(len(graph.get_iounits())), 2))
                    for combination in combinations:
                        graph_copy = deepcopy(graph)
                        ios = graph_copy.get_iounits()
                        selected_ios = [ios[combination[0]], ios[combination[1]]]
                        operation = tio(selected_ios)
                        graph_copy.add_operation([operation])
                        nodes_to_visit.append((deepcopy(graph_copy), node[1] + 1))

    # Use pickle to serialize the results for future use
    with open("../data/{}input{}degree.pkl".format(input_num, degree), 'wb') as f:
        pickle.dump(possible_graphs, f, pickle.HIGHEST_PROTOCOL)

    return possible_graphs


if __name__ == "__main__":
    pgs = exhaust_graph(degree=5, graphs_needed=1000000)
    print("{} possible graphs in total".format(len(pgs)))
