from legacy.pseq.graph import *


class Transformation(ABC):
    def __init__(self, graph, trans_arg=None, trans_type=None):
        self.graph = graph
        self.trans_type = trans_type
        self.trans_arg = trans_arg

    @abstractmethod
    def transform(self):
        raise NotImplementedError()

    @abstractmethod
    def satisfy_pattern(self, *args):
        raise NotImplementedError()


class ArithmeticMeanGeometricMean(Transformation):
    def __init__(self, graph, operation):
        if self.satisfy_pattern(operation) is False:
            raise InputError("Input operation is of type {}".format(operation.operation_type),
                             "while it should be an addition")
        Transformation.__init__(self, graph, trans_type="AMGM")
        self.operation = operation

    def satisfy_pattern(self, operation):
        if operation.operation_type == "Addition":
            return True
        else:
            return False

    def transform(self):
        two = [IOunit(Symbol(name="two", value=2, expression="2"))]
        mul1 = Multiplication(self.operation.inputs)
        sqrt1 = SquareRoot(mul1.output)
        mul2 = Multiplication(sqrt1.output + two)
        self.operation.get_output()[0].content = mul2.get_output()[0].content
        ind = self.graph.operations.index(self.operation)
        self.graph.operations[ind:ind + 1] = [mul1, sqrt1, mul2]
        for inp in self.operation.inputs:
            del inp.forward_ops[inp.forward_ops.index(self.operation)]


class Swap(Transformation):
    def __init__(self, graph, operations, swap_configuration: int):
        """

        :param graph: Graph to perform swap on
        :param operations: Operations to perform swap on, must be of the same type and be associative
        :param swap_configuration: Configuration of the swap result. For a graph ((x + y) + z), configuration 0 gives
        ((x + z) + y) and configuration 1 gives ((y + z) + x)
        """
        self.operations = operations
        self.upstream_op, self.downstream_op = None, None
        satisfied = self.satisfy_pattern(operations)
        if satisfied is False:
            raise InputError("Input operation is of type {}".
                             format(operations[0].operation_type + " " + operations[1].operation_type),
                             "while there are no valid swaps to perform")
        Transformation.__init__(self, graph, trans_type="Swap", trans_arg=swap_configuration)

    def satisfy_pattern(self, operations: List[Operation]) -> bool:
        # Only swap two operations at a time
        if len(operations) != 2:
            return False
        # The type of the two operations must be the same
        if operations[0].operation_type != operations[1].operation_type:
            return False
        # Operations must be associative in order to be swapped
        for operation in operations:
            if operation.operation_type in one_input_ops or operation.operation_type not in associative_ops:
                return False
        # One operation upstream and another downstream
        if operations[1] is operations[0].get_output()[0].forward_ops[0]:
            self.upstream_op = operations[0]
            self.downstream_op = operations[1]
        elif operations[0] is operations[1].get_output()[0].forward_ops[0]:
            self.upstream_op = operations[1]
            self.downstream_op = operations[0]
        else:
            return False
        return True

    def transform(self):
        inputs = self.upstream_op.inputs + self.downstream_op.inputs
        inputs.remove(self.upstream_op.get_output()[0])

        # Reconstruct links from iounits to operations
        inputs[1 - self.trans_arg].forward_ops.remove(self.upstream_op)
        inputs[1 - self.trans_arg].forward_ops.append(self.downstream_op)
        inputs[2].forward_ops.remove(self.downstream_op)
        inputs[2].forward_ops.append(self.upstream_op)
        # Reconfigure inputs of the operations
        self.upstream_op.inputs[1 - self.trans_arg] = inputs[2]
        idx = self.downstream_op.inputs.index(inputs[2])
        self.downstream_op.inputs[idx] = inputs[1 - self.trans_arg]


class Partition(Transformation):
    def __init__(self, graph, iounit: IOunit, partition_weights: List[float]):
        if self.satisfy_pattern(partition_weights) is False:
            raise InputError("The partition weights are {}".format(partition_weights),
                             "The partition is invalid.")
        self.iounit = iounit
        Transformation.__init__(self, graph, trans_arg=partition_weights, trans_type="Partition")

    def satisfy_pattern(self, weights):
        if len(weights) == 2 and sum(weights) == 1:
            return True
        return False

    def transform(self):
        downstream_ops = [op for op in self.iounit.forward_ops]
        if downstream_ops:
            self.iounit.forward_ops = []
            weight1 = IOunit(Symbol(name="partition_weight_1", value=self.trans_arg[0],
                                    expression="( " + str(self.trans_arg[0]) + " )"))
            weight2 = IOunit(Symbol(name="partition_weight_2", value=self.trans_arg[1],
                                    expression="( " + str(self.trans_arg[1]) + " )"))
            mul1 = Multiplication([self.iounit, weight1])
            mul2 = Multiplication([self.iounit, weight2])
            add = Addition(mul1.get_output() + mul2.get_output())
            new_iounit = add.get_output()[0]
            new_iounit.forward_ops = downstream_ops
            for op in downstream_ops:
                op.inputs[op.inputs.index(self.iounit)] = new_iounit
            # Find the index in all operations from where to insert new operations in the graph
            insertion_idx = self.graph.operations.index(downstream_ops[0])
            self.graph.operations[insertion_idx:insertion_idx] = [mul1, mul2, add]
        else:
            # The iounit selected is the root node of the computation tree
            weight1 = IOunit(Symbol(name="partition_weight_1", value=self.trans_arg[0],
                                    expression="( " + str(self.trans_arg[0]) + " )"))
            weight2 = IOunit(Symbol(name="partition_weight_2", value=self.trans_arg[1],
                                    expression="( " + str(self.trans_arg[1]) + " )"))
            mul1 = Multiplication([self.iounit, weight1])
            mul2 = Multiplication([self.iounit, weight2])
            add = Addition(mul1.get_output() + mul2.get_output())
            new_iounit = add.get_output()[0]
            self.graph.add_operation([mul1, mul2, add])


if __name__ == "__main__":
    # Swap x+y+z several times
    x = [IOunit(Symbol("x", value=1., expression="x"))]
    y = [IOunit(Symbol("y", value=2., expression="y"))]
    z = [IOunit(Symbol("z", value=3., expression="z"))]
    G = Graph("testExpression", inputs=x + y + z)

    add1 = Addition(x + y)
    add2 = Addition(add1.get_output() + z)
    mul = Multiplication(add2.get_output() + z)

    G.add_operation([add1, add2, mul])
    print(G.g_evaluate_output()[0].content.value)
    print(G.g_expression())

    par = Partition(G, x[0], [0.3, 0.7])
    par.transform()
    print(G.g_evaluate_output()[0].content.value)
    print(G.g_expression())

    swap = Swap(G, operations=[add1, add2], swap_configuration=1)
    swap.transform()
    print(G.g_evaluate_output()[0].content.value)
    print(G.g_expression())

    amgm = ArithmeticMeanGeometricMean(G, operation=[op for op in G.operations if op.operation_type == "Addition"][2])
    amgm.transform()
    print(G.g_evaluate_output()[0].content.value)
    print(G.g_expression())
