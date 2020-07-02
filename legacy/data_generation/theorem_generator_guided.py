from legacy.logic_math import real_number_axioms
from logic.logic import Entity
from logic.utils import standard_numerical_functions


def expand(entity):
    if entity.recent_numerical_function.name == "add":
        operations = list()
        for ent in entity.rnc_operands:
            operations.extend(expand(ent))
    elif entity.recent_numerical_function.name == "mul":
        lhs, rhs = entity.rnc_operands
        if rhs.recent_numerical_function.name == "add":
            return [[real_number_axioms["AdditionMultiplicationDistribution"], [lhs] + rhs.rnc_operands]]
        elif lhs.recent_numerical_function.name == "add":
            return [
                [real_number_axioms["MultiplicationCommutativity"], [lhs, rhs]],
                [real_number_axioms["AdditionMultiplicationDistribution"], [rhs] + lhs.rnc_operands]
            ]
        else:
            return []
    elif entity.recent_numerical_function.name == "sqr":
        return [[real_number_axioms["SquareDefinition"], entity.rnc_operands]]
    else:
        raise NotImplementedError


if __name__ == "__main__":
    a = Entity(name="input1")
    b = Entity(name="input2")
    c = Entity(name="input3")
    a_and_b = standard_numerical_functions["add"].execute_nf([a, b])
    b_and_c = standard_numerical_functions["add"].execute_nf([b, c])
    c_and_a = standard_numerical_functions["add"].execute_nf([c, a])
    a_sqr = standard_numerical_functions["sqr"].execute_nf([a])
    b_sqr = standard_numerical_functions["sqr"].execute_nf([b])
    c_sqr = standard_numerical_functions["sqr"].execute_nf([c])
    zero = Entity(name="0", is_constant=True)
    one = Entity(name="1", is_constant=True)
    entities = [a, b, c, a_and_b, b_and_c, c_and_a, a_sqr, b_sqr, c_sqr, zero, one]
