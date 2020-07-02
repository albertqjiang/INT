import pickle
import random

from legacy.pseq.transformation import ArithmeticMeanGeometricMean
from legacy.pseq import Sequence

random.seed(1)


def apply_amgm_to_first_or_second(graph_path, expression_path):
    with open(graph_path, "rb") as fh:
        graphs = pickle.load(fh)

    proof_sequences = []

    for graph in graphs:
        additions = [op for op in graph.get_operations() if op.operation_type == "Addition"]
        if len(additions) != 2:
            continue

        position_to_apply_addition = random.randint(0, 1)
        addition_to_transform = additions[position_to_apply_addition]
        amgm = ArithmeticMeanGeometricMean(graph, addition_to_transform)
        seq = Sequence(graph)
        seq.add_transformation(amgm)

        expressions = seq.proof_seq[0].g_expression() + '\t' + seq.proof_seq[1].g_expression() \
                      + '\t' + str(position_to_apply_addition)
        proof_sequences.append(expressions)

    with open(expression_path, "w") as fh:
        for sentence in proof_sequences:
            fh.write(sentence)
            fh.write('\n')
        fh.close()


if __name__ == "__main__":
    graph_path = "../data/2input4degree.pkl"
    expression_path = "../data/first_or_second_addition_amgm_2_4.txt"
    apply_amgm_to_first_or_second(graph_path, expression_path)
