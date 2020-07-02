from logic.logic import Entity
from logic.utils import standard_logic_functions, standard_numerical_functions
from math.meta_theorem import MetaTheorem

from copy import deepcopy


class EquivalenceReflexibility(MetaTheorem):
    def __init__(self):
        input_no = 1
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(EquivalenceReflexibility, self).__init__(input_no=input_no,
                                                       assumption_size=assumption_size,
                                                       conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        a = a
        :param inputs: 1 input [a]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        assumptions = list()
        conclusions = [standard_logic_functions["Equivalent"].execute_lf([inputs[0], inputs[0]])]
        extra_entities = list()
        return {"Assumptions": assumptions, "Conclusions": conclusions}


class EquivalenceTransitivity(MetaTheorem):
    def __init__(self):
        input_no = 3
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(EquivalenceTransitivity, self).__init__(input_no=input_no,
                                                      assumption_size=assumption_size,
                                                      conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        If a = b and b = c, then a = c
        :param inputs: 3 inputs [a, b, c]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        assumptions = [standard_logic_functions["Equivalent"].execute_lf([inputs[0], inputs[1]]),
                       standard_logic_functions["Equivalent"].execute_lf([inputs[1], inputs[2]])]
        conclusions = [standard_logic_functions["Equivalent"].execute_lf([inputs[0], inputs[2]])]
        extra_entities = list()
        return {"Assumptions": assumptions, "Conclusions": conclusions}


class EquivalenceSymmetry(MetaTheorem):
    def __init__(self):
        input_no = 2
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(EquivalenceSymmetry, self).__init__(input_no=input_no,
                                                  assumption_size=assumption_size,
                                                  conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        If a = b, then b = a
        :param inputs: 2 inputs [a, b]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        assumptions = [standard_logic_functions["Equivalent"].execute_lf([inputs[0], inputs[1]])]
        conclusions = [standard_logic_functions["Equivalent"].execute_lf([inputs[1], inputs[0]])]
        extra_entities = list()
        return {"Assumptions": assumptions, "Conclusions": conclusions}


class EquivalenceSubstitution(MetaTheorem):
    def __init__(self):
        input_no = 2
        a = Entity(name="input1")
        b = Entity(name="input2")
        c = Entity(name="input3")
        a_sqr = standard_numerical_functions["sqr"].execute_nf([a])
        b_sqr = standard_numerical_functions["sqr"].execute_nf([b])
        a_mul_b = standard_numerical_functions["mul"].execute_nf([a, b])
        a_mul_b_add_a_mul_b = standard_numerical_functions["add"].execute_nf([a_mul_b, a_mul_b])
        a_sqr_add_b_sqr = standard_numerical_functions["add"].execute_nf([a_sqr, b_sqr])
        gt = standard_logic_functions["BiggerOrEqual"].execute_lf([a_sqr_add_b_sqr, a_mul_b_add_a_mul_b])
        result = self.execute_th([gt.ent_dic[6], c])
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(EquivalenceSubstitution, self).__init__(input_no=input_no,
                                                      assumption_size=assumption_size,
                                                      conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        If a = b, and gt(a) is true, then gt(b) is true
        :param inputs: 2 inputs [a, b]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        input_0 = inputs[0]
        input_1 = inputs[1]
        gt = inputs[0].root
        assumptions = [standard_logic_functions["Equivalent"].execute_lf([input_0, input_1])]

        gt_equivalent = deepcopy(gt)
        gt_equivalent.indexing()
        parent_node = gt_equivalent.ent_dic[input_0.parent_index]
        for ind, operand in enumerate(parent_node.operands):
            if operand.index == input_0.index:
                replace_ind = ind
                parent_node.operands[replace_ind] = input_1
        gt_equivalent.update_name()

        conclusions = [gt_equivalent]
        return {"Assumptions": assumptions, "Conclusions": conclusions, "ExtraEntities": []}


class AdditionCommutativity(MetaTheorem):
    def __init__(self):
        input_no = 2
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(AdditionCommutativity, self).__init__(input_no=input_no,
                                                    assumption_size=assumption_size,
                                                    conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        a + b = b + a
        :param inputs: 2 inputs [a, b]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        a_and_b = standard_numerical_functions["add"].execute_nf([inputs[0], inputs[1]])
        b_and_a = standard_numerical_functions["add"].execute_nf([inputs[1], inputs[0]])
        assumptions = list()
        conclusions = [standard_logic_functions["Equivalent"].execute_lf([a_and_b, b_and_a])]
        extra_entities = [a_and_b, b_and_a]
        return {"Assumptions": assumptions, "Conclusions": conclusions}


class AdditionAssociativity(MetaTheorem):
    def __init__(self):
        input_no = 3
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(AdditionAssociativity, self).__init__(input_no=input_no,
                                                    assumption_size=assumption_size,
                                                    conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        a + ( b + c ) = ( a + b ) + c
        :param inputs: 3 inputs [a, b, c]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        b_and_c = standard_numerical_functions["add"].execute_nf([inputs[1], inputs[2]])
        lhs = standard_numerical_functions["add"].execute_nf([inputs[0], b_and_c])
        a_and_b = standard_numerical_functions["add"].execute_nf([inputs[0], inputs[1]])
        rhs = standard_numerical_functions["add"].execute_nf([a_and_b, inputs[2]])
        assumptions = list()
        conclusions = [standard_logic_functions["Equivalent"].execute_lf([lhs, rhs])]
        extra_entities = [b_and_c, a_and_b, lhs, rhs]
        return {"Assumptions": assumptions, "Conclusions": conclusions}


class AdditionIdentity(MetaTheorem):
    def __init__(self):
        input_no = 1
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(AdditionIdentity, self).__init__(input_no=input_no,
                                               assumption_size=assumption_size,
                                               conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        a + 0 = a
        :param inputs: 1 input [a]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        zero_entity = Entity(name="0", is_constant=True)
        a_and_zero = standard_numerical_functions["add"].execute_nf([inputs[0], zero_entity])
        assumptions = list()
        conclusions = [standard_logic_functions["Equivalent"].execute_lf([inputs[0], a_and_zero])]
        extra_entities = [zero_entity, a_and_zero]
        return {"Assumptions": assumptions, "Conclusions": conclusions}


class OppositeDefinition(MetaTheorem):
    def __init__(self):
        input_no = 1
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(OppositeDefinition, self).__init__(input_no=input_no,
                                                 assumption_size=assumption_size,
                                                 conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        a + (-a) = 0
        :param inputs: 1 input [a]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        zero_entity = Entity(name="0", is_constant=True)
        opp_a = standard_numerical_functions["opp"].execute_nf([inputs[0]])
        a_and_opp_a = standard_numerical_functions["add"].execute_nf([inputs[0], opp_a])
        assumptions = list()
        conclusions = [standard_logic_functions["Equivalent"].execute_lf([a_and_opp_a, zero_entity])]
        extra_entities = [a_and_opp_a, zero_entity]
        return {"Assumptions": assumptions, "Conclusions": conclusions}


class OppositeTwiceProperty(MetaTheorem):
    def __init__(self):
        input_no = 1
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(OppositeTwiceProperty, self).__init__(input_no=input_no,
                                                    assumption_size=assumption_size,
                                                    conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        -(-a) = a
        :param inputs: 1 input [a]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        opp_a = standard_numerical_functions["opp"].execute_nf([inputs[0]])
        opp_opp_a = standard_numerical_functions["opp"].execute_nf([opp_a])
        assumptions = list()
        conclusions = [standard_logic_functions["Equivalent"].execute_lf([inputs[0], opp_opp_a])]
        extra_entities = [opp_a, opp_opp_a]
        return {"Assumptions": assumptions, "Conclusions": conclusions}


class SubtractionDefinition(MetaTheorem):
    def __init__(self):
        input_no = 2
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(SubtractionDefinition, self).__init__(input_no=input_no,
                                                    assumption_size=assumption_size,
                                                    conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        a - b = a + (-b)
        :param inputs: 2 inputs [a, b]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        lhs = standard_numerical_functions["sub"].execute_nf([inputs[0], inputs[1]])
        opposite_b = standard_numerical_functions["opp"].execute_nf([inputs[1]])
        rhs = standard_numerical_functions["add"].execute_nf([inputs[0], opposite_b])
        assumptions = list()
        conclusions = [standard_logic_functions["Equivalent"].execute_lf([lhs, rhs])]
        extra_entities = [lhs, rhs]
        return {"Assumptions": assumptions, "Conclusions": conclusions}


class MultiplicationCommutativity(MetaTheorem):
    def __init__(self):
        input_no = 2
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(MultiplicationCommutativity, self).__init__(input_no=input_no,
                                                          assumption_size=assumption_size,
                                                          conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        a * b = b * a
        :param inputs: 2 inputs [a, b]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        lhs = standard_numerical_functions["mul"].execute_nf([inputs[0], inputs[1]])
        rhs = standard_numerical_functions["mul"].execute_nf([inputs[1], inputs[0]])
        assumptions = list()
        conclusions = [standard_logic_functions["Equivalent"].execute_lf([lhs, rhs])]
        extra_entities = [lhs, rhs]
        return {"Assumptions": assumptions, "Conclusions": conclusions}


class MultiplicationAssociativity(MetaTheorem):
    def __init__(self):
        input_no = 3
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(MultiplicationAssociativity, self).__init__(input_no=input_no,
                                                          assumption_size=assumption_size,
                                                          conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        a * ( b * c ) = ( a * b ) * c
        :param inputs: 3 inputs [a, b, c]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        b_times_c = standard_numerical_functions["mul"].execute_nf([inputs[1], inputs[2]])
        lhs = standard_numerical_functions["mul"].execute_nf([inputs[0], b_times_c])
        a_times_b = standard_numerical_functions["mul"].execute_nf([inputs[0], inputs[1]])
        rhs = standard_numerical_functions["mul"].execute_nf([a_times_b, inputs[2]])
        assumptions = list()
        conclusions = [standard_logic_functions["Equivalent"].execute_lf([lhs, rhs])]
        extra_entities = [b_times_c, a_times_b, lhs, rhs]
        return {"Assumptions": assumptions, "Conclusions": conclusions}


class MultiplyByZero(MetaTheorem):
    def __init__(self):
        input_no = 1
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(MultiplyByZero, self).__init__(input_no=input_no,
                                             assumption_size=assumption_size,
                                             conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        0 * a = 0
        :param inputs: 1 input [a]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        zero_entity = Entity(name="0", is_constant=True)
        lhs = standard_numerical_functions["mul"].execute_nf([zero_entity, inputs[0]])
        assumptions = list()
        conclusions = [standard_logic_functions["Equivalent"].execute_lf([lhs, zero_entity])]
        extra_entities = [zero_entity, lhs]
        return {"Assumptions": assumptions, "Conclusions": conclusions}


class MultiplicationIdentity(MetaTheorem):
    def __init__(self):
        input_no = 1
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(MultiplicationIdentity, self).__init__(input_no=input_no,
                                                     assumption_size=assumption_size,
                                                     conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        1 * a = a
        :param inputs: 1 input [a]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        multiplication_identity = Entity(name="1", is_constant=True)
        lhs = standard_numerical_functions["mul"].execute_nf([multiplication_identity, inputs[0]])
        assumptions = list()
        conclusions = [standard_logic_functions["Equivalent"].execute_lf([lhs, inputs[0]])]
        extra_entities = [multiplication_identity, lhs]
        return {"Assumptions": assumptions, "Conclusions": conclusions}


class AdditionMultiplicationDistribution(MetaTheorem):
    def __init__(self):
        input_no = 3
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(AdditionMultiplicationDistribution, self).__init__(input_no=input_no,
                                                                 assumption_size=assumption_size,
                                                                 conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        x * ( y + z ) = ( x * y ) + ( x * z )
        :param inputs: 3 inputs [x, y, z]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        y_and_z = standard_numerical_functions["add"].execute_nf([inputs[1], inputs[2]])
        lhs = standard_numerical_functions["mul"].execute_nf([inputs[0], y_and_z])
        x_times_y = standard_numerical_functions["mul"].execute_nf([inputs[0], inputs[1]])
        x_times_z = standard_numerical_functions["mul"].execute_nf([inputs[0], inputs[2]])
        rhs = standard_numerical_functions["add"].execute_nf([x_times_y, x_times_z])
        assumptions = list()
        conclusions = [standard_logic_functions["Equivalent"].execute_lf([lhs, rhs])]
        extra_entities = [y_and_z, lhs, x_times_y, x_times_z, rhs]
        return {"Assumptions": assumptions, "Conclusions": conclusions}


class InverseDefinition(MetaTheorem):
    def __init__(self):
        input_no = 1
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(InverseDefinition, self).__init__(input_no=input_no,
                                                assumption_size=assumption_size,
                                                conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        a * inv(a) = 1
        :param inputs: 1 input [a]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        inverse_a = standard_numerical_functions["inv"].execute_nf([inputs[0]])
        lhs = standard_numerical_functions["mul"].execute_nf([inputs[0], inverse_a])
        multiplication_identity = Entity(name="1", is_constant=True)
        assumptions = list()
        conclusions = [standard_logic_functions["Equivalent"].execute_lf([lhs, multiplication_identity])]
        extra_entities = [inverse_a, lhs, multiplication_identity]
        return {"Assumptions": assumptions, "Conclusions": conclusions}


class InequalityInverse(MetaTheorem):
    def __init__(self):
        input_no = 2
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(InequalityInverse, self).__init__(input_no=input_no,
                                                assumption_size=assumption_size,
                                                conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        If a >= b, then b <= a
        :param inputs: 2 inputs [a, b]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        assumptions = [standard_logic_functions["BiggerOrEqual"].execute_lf([inputs[0], inputs[1]])]
        conclusions = [standard_logic_functions["SmallerOrEqual"].execute_lf([inputs[1], inputs[0]])]
        extra_entities = list()
        return {"Assumptions": assumptions, "Conclusions": conclusions}


class DoubleInequalityImpliesEquivalence(MetaTheorem):
    def __init__(self):
        input_no = 2
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(DoubleInequalityImpliesEquivalence, self).__init__(input_no=input_no,
                                                                 assumption_size=assumption_size,
                                                                 conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        If a >= b and a <= b, then a = b
        :param inputs: 2 inputs [a, b]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        assumptions = [standard_logic_functions["BiggerOrEqual"].execute_lf([inputs[0], inputs[1]]),
                       standard_logic_functions["SmallerOrEqual"].execute_lf([inputs[0], inputs[1]])]
        conclusions = [standard_logic_functions["Equivalent"].execute_lf([inputs[0], inputs[1]])]
        extra_entities = list()
        return {"Assumptions": assumptions, "Conclusions": conclusions}


class EquivalenceImpliesDoubleInequality(MetaTheorem):
    def __init__(self):
        input_no = 2
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(EquivalenceImpliesDoubleInequality, self).__init__(input_no=input_no,
                                                                 assumption_size=assumption_size,
                                                                 conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        If a = b, then a >= b and a <= b
        :param inputs: 2 inputs [a, b]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        assumptions = [standard_logic_functions["Equivalent"].execute_lf([inputs[0], inputs[1]])]
        conclusions = [standard_logic_functions["BiggerOrEqual"].execute_lf([inputs[0], inputs[1]]),
                       standard_logic_functions["SmallerOrEqual"].execute_lf([inputs[0], inputs[1]])]
        extra_entities = list()
        return {"Assumptions": assumptions, "Conclusions": conclusions}


class InequalityTransitivity(MetaTheorem):
    def __init__(self):
        input_no = 3
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(InequalityTransitivity, self).__init__(input_no=input_no,
                                                     assumption_size=assumption_size,
                                                     conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        If a >= b and b >= c, then a >=c
        :param inputs: 3 inputs [a, b, c]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        assumptions = [standard_logic_functions["BiggerOrEqual"].execute_lf([inputs[0], inputs[1]]),
                       standard_logic_functions["BiggerOrEqual"].execute_lf([inputs[1], inputs[2]])]
        conclusions = [standard_logic_functions["BiggerOrEqual"].execute_lf([inputs[0], inputs[2]])]
        extra_entities = list()
        return {"Assumptions": assumptions, "Conclusions": conclusions}


class FirstPrincipleOfInequality(MetaTheorem):
    def __init__(self):
        input_no = 4
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(FirstPrincipleOfInequality, self).__init__(input_no=input_no,
                                                         assumption_size=assumption_size,
                                                         conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        If a >= b and c >= d, then a + c >= b + d
        :param inputs: 4 inputs [a, b, c, d]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        assumptions = [standard_logic_functions["BiggerOrEqual"].execute_lf([inputs[0], inputs[1]]),
                       standard_logic_functions["BiggerOrEqual"].execute_lf([inputs[2], inputs[3]])]
        a_and_c = standard_numerical_functions["add"].execute_nf([inputs[0], inputs[2]])
        b_and_d = standard_numerical_functions["add"].execute_nf([inputs[1], inputs[3]])
        conclusions = [standard_logic_functions["BiggerOrEqual"].execute_lf([a_and_c, b_and_d])]
        extra_entities = [a_and_c, b_and_d]
        return {"Assumptions": assumptions, "Conclusions": conclusions}


class SecondPrincipleOfInequality(MetaTheorem):
    def __init__(self):
        input_no = 3
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(SecondPrincipleOfInequality, self).__init__(input_no=input_no,
                                                          assumption_size=assumption_size,
                                                          conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        If a >= b and c >= 0, then a * c >= b * c
        :param inputs: 3 inputs [a, b, c]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        zero_entity = Entity(name="0", is_constant=True)
        a_times_c = standard_numerical_functions["mul"].execute_nf([inputs[0], inputs[2]])
        b_times_c = standard_numerical_functions["mul"].execute_nf([inputs[1], inputs[2]])
        assumptions = [standard_logic_functions["BiggerOrEqual"].execute_lf([inputs[0], inputs[1]]),
                       standard_logic_functions["BiggerOrEqual"].execute_lf([inputs[2], zero_entity])]
        conclusions = [standard_logic_functions["BiggerOrEqual"].execute_lf([a_times_c, b_times_c])]
        extra_entities = [zero_entity, a_times_c, b_times_c]
        return {"Assumptions": assumptions, "Conclusions": conclusions}


class SquareDefinition(MetaTheorem):
    def __init__(self):
        input_no = 1
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(SquareDefinition, self).__init__(input_no=input_no,
                                               assumption_size=assumption_size,
                                               conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        a ^ 2 = a * a
        :param inputs: 1 input [a]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        sqr_a = standard_numerical_functions["sqr"].execute_nf([inputs[0]])
        a_times_a = standard_numerical_functions["mul"].execute_nf([inputs[0], inputs[0]])
        assumptions = list()
        conclusions = [standard_logic_functions["Equivalent"].execute_lf([sqr_a, a_times_a])]
        extra_entities = [sqr_a, a_times_a]
        return {"Assumptions": assumptions, "Conclusions": conclusions}


class ConstantComparison(MetaTheorem):
    def __init__(self):
        input_no = 2
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(ConstantComparison, self).__init__(input_no=input_no,
                                                 assumption_size=assumption_size,
                                                 conclusion_size=conclusion_size)

    def execute_th(self, inputs):
        """
        Compare 2 constants
        :param inputs: 2 inputs [a, b]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        assumptions = list()
        extra_entities = list()
        if len(inputs) != 2 or (not inputs[0].is_constant) or (not inputs[1].is_constant):
            conclusions = list()
        elif float(inputs[0].name) >= float(inputs[1].name):
            conclusions = [standard_logic_functions["BiggerOrEqual"].execute_lf([inputs[0], inputs[1]])]
        else:
            conclusions = [standard_logic_functions["BiggerOrEqual"].execute_lf([inputs[1], inputs[0]])]
        return {"Assumptions": assumptions, "Conclusions": conclusions}


real_number_axioms = {
    "EquivalenceReflexibility": EquivalenceReflexibility(),
    "EquivalenceSymmetry": EquivalenceSymmetry(),
    "EquivalenceTransitivity": EquivalenceTransitivity(),
    "AdditionCommutativity": AdditionCommutativity(),
    "AdditionAssociativity": AdditionAssociativity(),
    "AdditionIdentity": AdditionIdentity(),
    "OppositeDefinition": OppositeDefinition(),
    "OppositeTwiceProperty": OppositeTwiceProperty(),
    "SubtractionDefinition": SubtractionDefinition(),
    "MultiplyByZero": MultiplyByZero(),
    "MultiplicationIdentity": MultiplicationIdentity(),
    "MultiplicationAssociativity": MultiplicationAssociativity(),
    "MultiplicationCommutativity": MultiplicationCommutativity(),
    "AdditionMultiplicationDistribution": AdditionMultiplicationDistribution(),
    "InverseDefinition": InverseDefinition(),
    "InequalityInverse": InequalityInverse(),
    "DoubleInequalityImpliesEquivalence": DoubleInequalityImpliesEquivalence(),
    "EquivalenceImpliesDoubleInequality": EquivalenceImpliesDoubleInequality(),
    "EquivalenceSubstitution": EquivalenceSubstitution(),
    "InequalityTransitivity": InequalityTransitivity(),
    "FirstPrincipleOfInequality": FirstPrincipleOfInequality(),
    "SecondPrincipleOfInequality": SecondPrincipleOfInequality(),
    "SquareDefinition": SquareDefinition(),
    "ConstantComparison": ConstantComparison(),
}


class AMGM(MetaTheorem):
    def __init__(self):
        input_no = 2
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        super(AMGM, self).__init__(input_no=input_no,
                                   assumption_size=assumption_size,
                                   conclusion_size=conclusion_size)

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
        return {"Assumptions": assumptions, "Conclusions": conclusions}


famous_theorems = {"AMGM": AMGM()}

# Merge two sets of theorems
complete_theorems = {**real_number_axioms, **famous_theorems}
