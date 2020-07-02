from legacy.pseq.operation import *


class Symbol:
    # Keep track of symbol names used
    symbol_names: List[str] = []

    def __init__(self, name, value=None, expression=None):
        Symbol.symbol_names.append(name)
        self.name = name
        self.value = value
        self.expression = expression

    @staticmethod
    def get_symbol_names():
        return Symbol.symbol_names


class IOunit:
    # Need to use a class to wrap up Input and Output units
    # Once a computation graph is set up, don't delete old IOunits, only alter their content
    # All inputs and outputs must be wrapped by an IOunit for easy tracking
    def __init__(self, symbol: Symbol, tensor=None) -> None:
        self.content = symbol
        self.forward_ops = []
        self.tensor = tensor


if __name__ == "__main__":
    x = [IOunit(Symbol(name="x", value=1, expression="x"))]
    y = [IOunit(Symbol(name="y", value=1, expression="y"))]

    added = Addition(x + y)
    print(x[0].forward_ops)
