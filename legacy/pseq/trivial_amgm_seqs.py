import random
from pprint import pprint


def random_amgm(g):
    """

    :param g: a graph
    :return: a sequence consisting of the original graph and one geometric_mean step
             None if the original graph contains no addition
    """

    additions = [operation for operation in g.operations if operation.operation_type == "Addition"]
    if len(additions) == 0:
        return None
    addition_to_transform = random.choice(additions)

    amgm = ArithmeticMeanGeometricMean(g, addition_to_transform)

    seq = Sequence(g)
    seq.add_transformation(amgm)

    return seq.proof_seq


if __name__ == "__main__":
    x = [IOunit(Symbol("x", value=1., expression="x"))]
    y = [IOunit(Symbol("y", value=4., expression="y"))]
    z = [IOunit(Symbol("z", value=9., expression="z"))]
    G = Graph("testExpression", inputs=[x, y, z])

    rec1 = Reciprocal(x)
    rec2 = Reciprocal(y)
    rec3 = Reciprocal(z)

    mul1 = Multiplication(x + rec2.get_output())
    mul2 = Multiplication(y + rec3.get_output())
    mul3 = Multiplication(z + rec1.get_output())

    add1 = Addition(mul1.output + mul2.output)
    add2 = Addition(add1.output + mul3.output)
    f = add2.output[0]

    G.add_operation([rec1, rec2, rec3, mul1, mul2, mul3, add1, add2])

    seq = random_amgm(G)

    print("The values of input symbols are:")
    pprint([s[0].content.expression + ": " + str(s[0].content.value) for s in [x, y, z]])
    print("The initial expression is", seq[0].g_expression())
    print("It evaluates to", seq[0].g_evaluate_output()[0].content.value)
    print("An AM-GM transformation is performed at a random valid place.")
    print("Now the expression of the graph is:", seq[-1].g_evaluate_output()[0].content.expression)
    print("It evaluates to", seq[-1].g_evaluate_output()[0].content.value)
    print("A proof sequence is recorded:")
    pprint(seq)
