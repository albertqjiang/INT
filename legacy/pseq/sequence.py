from legacy.pseq.graph import Graph
from legacy.pseq.transformation import *


class Sequence:
    def __init__(self, graph):
        self.proof_seq = [deepcopy(graph)]
        self.graph = graph

    def add_transformation(self, transformation):
        if transformation.graph is not self.graph:
            raise InputError("Sequence is for {} while transformation is for {}".format(
                self.graph.g_expression(), transformation.graph.g_expression()), "Transformation graph not matched.")

        # self.proof_seq.append(deepcopy(transformation))

        transformation.transform()

        self.proof_seq.append(deepcopy(self.graph))


if __name__ == "__main__":
    # Note: there are two parallel [expression, transformation ...] sequences
    # The real effective one is implicitly used for the graph attribute
    # What's in the proof_seq is merely a copy
    # Define basic elements
    x = Symbol("x", value=1., expression="x")
    y = Symbol("y", value=2., expression="y")
    G = Graph("testExpression", inputs=[x, y])
    seq = Sequence(G)

    # Define computation graph for G
    rec1 = Reciprocal([x])
    rec2 = Reciprocal([y])

    mul1 = Multiplication([x, rec2.output])
    mul2 = Multiplication([y, rec1.output])

    add1 = Addition([mul1.output, mul2.output])
    f = add1.output

    G.add_operation([rec1, rec2, mul1, mul2, add1])

    print(G.g_evaluate_output().value)

    # Check deepcopy
    print(seq.proof_seq[0] is G)

    # Define transformation
    amgm1 = ArithmeticMeanGeometricMean(G, add1)
    seq.add_transformation(amgm1)

    print(seq.proof_seq)

    print(seq.proof_seq[-1].g_evaluate_output().value)

    print(seq.proof_seq[-1].g_expression())

    print(seq.proof_seq[-1] is G)
