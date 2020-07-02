__author__ = "Albert Jiang"

from legacy.pseq.symbol import *
from legacy.pseq import InputError


class Operation:
    def __init__(self, inputs, operation_type):
        """

        :param inputs: a list of IOunits
        :param operation_type: the type of the operation, specified in each subclass
        """
        for inp in inputs:
            inp.forward_ops.append(self)
        self.inputs = inputs
        self.operation_type = operation_type

    def get_output(self):
        """

        :return: shall return a list of IOunits
        """
        pass


class Combination(Operation):
    # Keep track of Combinations used
    combinations = 0

    def __init__(self, inputs):
        Operation.__init__(self, inputs, operation_type="Combination")
        Combination.combinations += 1
        self.operations = []

    def add_operation(self, operation):
        self.operations.append(operation)

    def evaluate(self):
        try:
            for o in self.operations:
                o.evaluate()
        except IndexError:
            print("The operation array is empty.")

    def get_output(self):
        return self.operations[-1].output

    @staticmethod
    def get_num_of_combinations():
        return Combination.combinations


class Addition(Operation):
    # Keep track of additions used
    additions = 0

    def __init__(self, inputs):
        if len(inputs) != 2:
            raise InputError("Input is {}".format(inputs), "Addition only deals with two inputs.")
        Operation.__init__(self, inputs, operation_type="Addition")
        Addition.additions += 1
        outs = Symbol(name="added{}".format(Addition.additions),
                      expression="( " + self.inputs[0].content.expression + " + " + self.inputs[
                          1].content.expression + " )")
        self.output = [IOunit(outs)]

    def evaluate(self):
        try:
            self.output[0].content.value = self.inputs[0].content.value + self.inputs[1].content.value
            self.output[0].content.expression = "( " + self.inputs[0].content.expression + " + " + self.inputs[
                1].content.expression + " )"
        except TypeError:
            print("Values added({}) are invalid.".format([i.content.value for i in self.inputs]))

    def update_expression(self):
        self.output[0].content.expression = "( " + self.inputs[0].content.expression + " + " + self.inputs[
            1].content.expression + " )"

    def get_output(self):
        return self.output

    @staticmethod
    def get_num_of_additions():
        return Addition.additions


class Multiplication(Operation):
    # Keep track of multiplications used
    multiplications = 0

    def __init__(self, inputs):
        if len(inputs) != 2:
            raise InputError("Input is {}".format(inputs), "Multiplication only deals with two inputs.")
        Operation.__init__(self, inputs, operation_type="Multiplication")
        Multiplication.multiplications += 1
        outs = Symbol(name="multiplied{}".format(Multiplication.multiplications),
                      expression=self.inputs[0].content.expression + " * " + self.inputs[1].content.expression)
        self.output = [IOunit(outs)]

    def evaluate(self):
        try:
            self.output[0].content.value = self.inputs[0].content.value * self.inputs[1].content.value
            self.output[0].content.expression = \
                self.inputs[0].content.expression + " * " + self.inputs[1].content.expression
        except TypeError:
            print("Values multiplied({}) are invalid.".format([i.content.value for i in self.inputs]))

    def update_expression(self):
        self.output[0].content.expression = \
            self.inputs[0].content.expression + " * " + self.inputs[1].content.expression

    def get_output(self):
        return self.output

    @staticmethod
    def get_num_of_multiplications():
        return Multiplication.multiplications


class SquareRoot(Operation):
    # Keep track of square roots used
    square_roots = 0

    def __init__(self, inputs):
        Operation.__init__(self, inputs, operation_type="SquareRoot")
        if len(inputs) != 1:
            raise InputError("Input is {}".format(inputs), "SquareRoot only deals with one input.")
        elif (inputs[0].content.value is not None) and inputs[0].content.value < 0:
            raise InputError("Input is {}".format(inputs), "SquareRoot must be applied to positive values only.")
        SquareRoot.square_roots += 1
        outs = Symbol(name="squarerooted{}".format(SquareRoot.square_roots),
                      expression="sqrt ( " + self.inputs[0].content.expression + " )")
        self.output = [IOunit(outs)]

    def evaluate(self):
        try:
            self.output[0].content.value = math.sqrt(self.inputs[0].content.value)
            self.output[0].content.expression = "sqrt ( " + self.inputs[0].content.expression + " )"
        except TypeError:
            print("Value squarerooted is invalid.")

    def update_expression(self):
        self.output[0].content.expression = "sqrt ( " + self.inputs[0].content.expression + " )"

    def get_output(self):
        return self.output

    @staticmethod
    def get_num_of_sqrts():
        return SquareRoot.square_roots


class Reciprocal(Operation):
    # Keep track of Reciprocals used
    reciprocals = 0

    def __init__(self, inputs):
        if len(inputs) != 1:
            raise InputError("Input is {}".format(inputs), "Reciprocal only deals with one input.")
        Operation.__init__(self, inputs, operation_type="Reciprocal")
        Reciprocal.reciprocals += 1
        outs = Symbol(name="Reciprocal{}".format(Reciprocal.reciprocals),
                      expression="1/ ( " + self.inputs[0].content.expression + " )")
        self.output = [IOunit(outs)]

    def evaluate(self):
        try:
            self.output[0].content.value = 1 / self.inputs[0].content.value
            self.output[0].content.expression = "1/ ( " + self.inputs[0].content.expression + " )"
        except TypeError:
            print("Value inverted is invalid.")
        except ZeroDivisionError:
            print("Cannot divide by zero.")

    def update_expression(self):
        self.output[0].content.expression = "1/ ( " + self.inputs[0].content.expression + " )"

    def get_output(self):
        return self.output

    @staticmethod
    def get_num_of_reciprocals():
        return Reciprocal.reciprocals


if __name__ == "__main__":
    x = IOunit(Symbol(name="x", expression="( x )", value=3))
    y = IOunit(Symbol(name="y", expression="( y )", value=2))
    add1 = Addition([x, y])
    add2 = Addition([x, y])
    mul = Multiplication(add1.output + add2.output)
    add1.evaluate()
    add2.evaluate()
    mul.evaluate()
    print(mul.get_output()[0].content.value)
    print(mul.get_output()[0].content.expression)
    print(mul.get_output()[0].content.name)

    comb = Combination([x, y])
    comb.add_operation(add1)
    comb.add_operation(add2)
    comb.add_operation(mul)
    print(comb.get_output()[0].content.expression)
    print(comb.get_output()[0].content.value)
