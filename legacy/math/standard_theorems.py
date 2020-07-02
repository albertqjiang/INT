from logic.logic import Entity
from logic.utils import standard_logic_functions, standard_numerical_functions
from legacy.logic_math import MetaTheorem


class SecondPrincipleOfInequality(MetaTheorem):
    def __init__(self):
        super(SecondPrincipleOfInequality, self).__init__(input_no=3)

    def execute_th(self, inputs):
        """
        If a >= b, a-c >= b-c.
        :param inputs: 3 inputs, [a, b, c]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        assumptions = standard_logic_functions["BiggerOrEqual"].execute_lf([inputs[0], inputs[1]])
        a_minus_c = standard_numerical_functions["sub"].execute_nf([inputs[0], inputs[2]])
        b_minus_c = standard_numerical_functions["sub"].execute_nf([inputs[1], inputs[2]])
        conclusions = [standard_logic_functions["BiggerOrEqual"].execute_lf([a_minus_c, b_minus_c])]
        return {"Assumptions": assumptions, "Conclusions": conclusions, "ExtraEntities": [a_minus_c, b_minus_c]}


class AMGM(MetaTheorem):
    def __init__(self):
        super(AMGM, self).__init__(input_no=2)

    def execute_th(self, inputs):
        """
        If a, b >= 0, then a + b >= 2 * sqrt(ab)
        :param inputs: 2 inputs, [a, b]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        zero_entity = Entity(name="0", is_constant=True)
        two_entity = Entity(name="2", is_constant=True)
        assumptions = [standard_logic_functions["BiggerOrEqual"].execute_lf([inputs[0], zero_entity]),
                       standard_logic_functions["BiggerOrEqual"].execute_lf([inputs[1], zero_entity])]
        a_and_b = standard_numerical_functions["add"].execute_nf([inputs[0], inputs[1]])
        a_time_b = standard_numerical_functions["mul"].execute_nf([inputs[0], inputs[1]])
        sqrt_ab = standard_numerical_functions["sqrt"].execute_nf([a_time_b])
        two_sqrt_ab = standard_numerical_functions["mul"].execute_nf([two_entity, sqrt_ab])
        conclusions = [standard_logic_functions["BiggerOrEqual"].execute_lf([a_and_b, two_sqrt_ab])]
        extra_entities = [a_and_b, a_time_b, sqrt_ab, two_sqrt_ab]
        return {"Assumptions": assumptions, "Conclusions": conclusions, "ExtraEntities": extra_entities}


class SquareNonNegative(MetaTheorem):
    def __init__(self):
        super(SquareNonNegative, self).__init__(input_no=1)

    def execute_th(self, inputs):
        """
        a ^ 2 >= 0
        :param inputs: 1 input, [a]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        zero_entity = Entity(name="0", is_constant=True)
        a_squared = standard_numerical_functions["sqr"].execute_nf([inputs[0]])
        assumptions = list()
        conclusions = [standard_logic_functions["BiggerOrEqual"].execute_lf([a_squared, zero_entity])]
        extra_entities = [a_squared]
        return {"Assumptions": assumptions, "Conclusions": conclusions, "ExtraEntities": extra_entities}


class SubtractionInverseAdditions(MetaTheorem):
    def __init__(self):
        super(SubtractionInverseAdditions, self).__init__(input_no=3)

    def execute_th(self, inputs):
        """
        If b = c, then a + b -c = a
        :param inputs: 3 inputs: [a, b, c]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        assumptions = [standard_logic_functions["Equivalent"].execute_lf([inputs[1], inputs[2]])]
        a_and_b = standard_numerical_functions["add"].execute_nf([inputs[0], inputs[1]])
        a_and_b_minus_c = standard_numerical_functions["sub"].execute_nf([a_and_b, inputs[2]])
        conclusions = [standard_logic_functions["Equivalent"].execute_lf([a_and_b_minus_c, inputs[0]])]
        extra_entities = [a_and_b_minus_c]
        return {"Assumptions": assumptions, "Conclusions": conclusions, "ExtraEntities": extra_entities}


standard_theorems = {
    "SecondPrincipleOfInequality": SecondPrincipleOfInequality(),
    "AMGM": AMGM(),
    "SquareNonNegative": SquareNonNegative(),
    "SubtractionInverseAddition": SubtractionInverseAdditions(),
}
