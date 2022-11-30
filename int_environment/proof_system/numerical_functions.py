from logic.logic import NumericalFunction
from collections import OrderedDict

add = NumericalFunction("add", 2)
opp = NumericalFunction("opp", 1)
mul = NumericalFunction("mul", 2)
sqr = NumericalFunction("sqr", 1)
inv = NumericalFunction("inv", 1)
necessary_numerical_functions = \
    OrderedDict(
        [("add", add),
         ("opp", opp),
         ("mul", mul),
         ("sqr", sqr),
         ("inv", inv)]
    )
