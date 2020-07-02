import pickle

from legacy.pseq.transformation import ArithmeticMeanGeometricMean
from legacy.pseq import Sequence


def apply_amgm():
    with open("../data/2input5degree.pkl", "rb") as fh:
        graphs = pickle.load(fh)

    proof_sequences = []

    for graph in graphs:
        additions = [op for op in graph.get_operations() if op.operation_type == "Addition"]
        if not additions:
            continue
        first_addition = additions[0]
        amgm = ArithmeticMeanGeometricMean(graph, first_addition)
        seq = Sequence(graph)
        seq.add_transformation(amgm)
        if seq.proof_seq[1].g_expression().find('sqrt') == -1:
            original_graph = seq.proof_seq[0]
            bugged_graph = seq.proof_seq[1]
            bugged_operations = bugged_graph.operations
        expressions = seq.proof_seq[0].g_expression() + '\t' + seq.proof_seq[1].g_expression()
        proof_sequences.append(expressions)

    with open("../data/first_addition_amgm_2_5.txt", "w") as fh:
        for pair in proof_sequences:
            fh.write(pair)
            fh.write('\n')
        fh.close()


if __name__ == "__main__":
    apply_amgm()
