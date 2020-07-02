import pickle
import random

from legacy.pseq import InputError
from legacy.pseq.transformation import ArithmeticMeanGeometricMean, Partition

possible_transformation_types = ["AMGM", "Partition"]


def get_sublist_position(whole_list, sub_list):
    length = len(sub_list)
    for i in range(0, len(whole_list) - length + 1):
        if whole_list[i:i + length] == sub_list:
            return i, i + length


def random_transformation(graph, transformation_type):
    if transformation_type == "AMGM":
        all_operations = [op for op in graph.operations if op.operation_type == "Addition"]
        if all_operations:
            operation = random.choice(all_operations)
            out_words = operation.get_output()[0].content.expression
            positions = get_sublist_position(graph.g_expression().split(" "), out_words.split(" "))
            # print("geometric_mean applied to: ", out_words.split(" "))
            # print("graph expression: ", graph.g_expression().split(" "))
            # print(get_sublist_position(graph.g_expression().split(" "), out_words.split(" ")))
            trans = ArithmeticMeanGeometricMean(graph, operation)
            trans.transform()
            return graph, transformation_type, positions
        return None

    elif transformation_type == "Partition":
        io_subject = random.choice(graph.get_iounits())
        out_words = io_subject.content.expression
        positions = get_sublist_position(graph.g_expression().split(" "), out_words.split(" "))
        trans = Partition(graph, iounit=io_subject, partition_weights=[0.5, 0.5])
        trans.transform()
        return graph, transformation_type, positions

    else:
        raise InputError("Transformation type required is {}".format(transformation_type), "This isn't valid.")


def generate_seq_strings(input=2, degree=5, depth=1):
    file = "../data/{}input{}degree.pkl".format(input, degree)
    graphs = pickle.load(open(file, "rb"))
    with open("../data/{}input{}degree{}depth_trans_chain.txt".format(input, degree, depth), "w") as fh:
        for graph in graphs:
            sequence_string = ""
            for _ in range(depth):
                transformation_type = random.choice(possible_transformation_types)
                original_string = graph.g_expression()
                place_holder = random_transformation(graph, transformation_type)
                if place_holder:
                    graph, t_type, positions = place_holder[0], place_holder[1], place_holder[2]
                    sequence_string += original_string + '\t' + t_type + ':' + str(positions) + '\n'
                else:
                    continue
            if '\n' in sequence_string:
                sequence_string += graph.g_expression() + ';\n'
                fh.write(sequence_string)


if __name__ == "__main__":
    generate_seq_strings(degree=5, depth=3)
