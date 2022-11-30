import itertools


from logic.logic import *

Real = LogicFunction("Real", 1)
NonNegative = LogicFunction("NonNegative", 1)
BiggerOrEqual = LogicFunction("BiggerOrEqual", 2)
SmallerOrEqual = LogicFunction("SmallerOrEqual", 2)
Equivalent = LogicFunction("Equivalent", 2)

standard_logic_functions = {
    "Real": Real,
    "NonNegative": NonNegative,
    "BiggerOrEqual": BiggerOrEqual,
    "SmallerOrEqual": SmallerOrEqual,
    "Equivalent": Equivalent,
}

add = NumericalFunction("add", 2)
opp = NumericalFunction("opp", 1)
sub = NumericalFunction("sub", 2)
mul = NumericalFunction("mul", 2)
sqr = NumericalFunction("sqr", 1)
sqrt = NumericalFunction("sqrt", 1)
inv = NumericalFunction("inv", 1)
geometric_mean = NumericalFunction("geometric_mean", 2)
identity = NumericalFunction("identity", 1)
standard_numerical_functions = {
    "add": add,
    "opp": opp,
    "sub": sub,
    "mul": mul,
    "sqr": sqr,
    "sqrt": sqrt,
    "inv": inv,
    "geometric_mean": geometric_mean,
    "identity": identity,
}

real_sqr_non_neg = Theorem(name="real_sqr_non_neg", input_no=2, input_constraints=[(sqr, (0,), (1,))],
                           assumptions=[(Real, (0,))], conclusions=[(NonNegative, (1,))])
amgm = Theorem(name="AMGM for 2 elements", input_no=4,
               input_constraints=[(add, (0, 1), (2, )), (geometric_mean, (0, 1), (3, ))],
               assumptions=[(NonNegative, (0, )), (NonNegative, (1, ))],
               conclusions=[(BiggerOrEqual, (2, 3))])

standard_theorems = {
    "real_sqr_non_neg": real_sqr_non_neg,
    "amgm": amgm,
}


class ReplayBuffer:
    def __init__(self, name, max_size):
        self.buffer = list()
        self.name = name
        self.max_size = max_size
        import random

    def cache(self, other):
        self.buffer.extend(other)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]
        random.shuffle(self.buffer)

    def sample(self, sample_size):
        return random.sample(self.buffer, sample_size)


def set_random_seed(seed):
    import torch
    import random
    torch.manual_seed(seed)
    random.seed(seed)


def element_index_in_list_according_to_name(list_of_interest, element):
    for i, item in enumerate(list_of_interest):
        if item.name == element.name:
            return i


def smaller(a, b):
    if a <= b:
        return a
    return b


def exhaust_actions(prover):
    all_actions = list()
    theorems = prover.proof.lemmas
    entities = prover.proof.entities
    for theorem in theorems:
        entity_combinations = itertools.combinations(entities, r=theorem.input_no)
        for operands in entity_combinations:
            action = {"action_type": "theorem", "action": [theorem, operands]}
            all_actions.append(action)
    return all_actions
