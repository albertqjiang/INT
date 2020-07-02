from pprint import pprint
import numpy as np

from legacy.pseq.transformation import *


class Graph:
    def __init__(self, name, inputs, output=None):
        self.name = name
        self.inputs = inputs
        self.iounits = list()
        for inp in inputs:
            self.iounits.append(inp)
        self.operations = []
        self.output = output

    def set_input_values(self, num_tuple):
        if len(self.inputs) != len(num_tuple):
            raise InputError("Input length {} does not equal vector length {}".format(len(self.inputs), len(num_tuple)))
        for i in range(len(self.inputs)):
            self.inputs[i].content.value = num_tuple[i]

    def add_operation(self, operations):
        for operation in operations:
            self.operations.append(operation)
            self.iounits.append(operation.get_output()[0])

    def replace_operation(self, old_op, new_op):
        self.operations[self.operations.index(old_op)] = new_op

    def g_evaluate_output(self):
        for o in self.operations:
            o.evaluate()
        return self.operations[-1].get_output()

    def g_batch_evaluate(self, tuples):
        evaluation = np.zeros((1, tuples.shape[0]))
        for i in range(len(tuples)):
            self.set_input_values(tuples[i])
            evaluation[0][i] = self.g_evaluate_value()
        return evaluation

    def g_evaluate_value(self):
        return self.g_evaluate_output()[0].content.value

    def g_expression(self):
        for o in self.operations:
            o.update_expression()
        return self.operations[-1].get_output()[0].content.expression

    def get_iounits(self):
        return self.iounits

    def get_operations(self):
        return self.operations

    def get_num_of_leaves(self):
        """

        :return: the number of leaf symbols in the graph
        """
        count = 0
        for iounit in self.iounits:
            if len(iounit.forward_ops) == 0:
                count += 1
        return count


if __name__ == "__main__":
    # Transform x/y + y/x to 2
    x = [IOunit(Symbol("x", value=1., expression="x"))]
    y = [IOunit(Symbol("y", value=2., expression="y"))]
    G = Graph("testExpression", inputs=x + y)

    rec1 = Reciprocal(x)
    rec2 = Reciprocal(y)

    mul1 = Multiplication(x + rec2.output)
    mul2 = Multiplication(y + rec1.output)

    add1 = Addition(mul1.output + mul2.output)

    G.add_operation([rec1, rec2, mul1, mul2, add1])

    print(G.g_evaluate_output()[0].content.value)

    amgm1 = ArithmeticMeanGeometricMean(G, add1)
    amgm1.transform()

    print(G.g_evaluate_output()[0].content.value)

    x[0].content.value = 7823.

    print(G.g_evaluate_output()[0].content.value)
    print(G.g_expression())
    pprint([io.content.expression for io in G.iounits])
    print(G.get_num_of_leaves())
