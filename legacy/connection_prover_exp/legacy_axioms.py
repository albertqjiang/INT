import random

from proof_system.logic_functions import necessary_logic_functions
from proof_system.meta_axiom import MetaAxiom
from proof_system.numerical_functions import necessary_numerical_functions
from proof_system.utils import search_operator_operands_in_gt
from logic.logic import Entity


class InequalityTransitivity(MetaAxiom):
    def __init__(self):
        equivalence_theorem = False
        input_no = 3
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        assumption_types = [assump.logic_function.name for assump in result["Assumptions"]]
        super(InequalityTransitivity, self).__init__(input_no=input_no,
                                                     assumption_size=assumption_size,
                                                     conclusion_size=conclusion_size,
                                                     equivalence_theorem=equivalence_theorem,
                                                     assumption_types=assumption_types)

    def execute_th(self, operands, mode="generate"):
        """
        If a >= b and b >= c, then a >= c
        :param operands: 3 operands [a, b, c]
        :return: dict(Assumptions, Conclusions and ExtraEntities)
        """
        a, b, c, = operands
        a_copied1, a_copied2 = a, a
        b_copied1, b_copied2 = b, b
        c_copied1, c_copied2 = c, c
        assumptions = [necessary_logic_functions["BiggerOrEqual"].execute_lf([a_copied1, b_copied1]),
                       necessary_logic_functions["BiggerOrEqual"].execute_lf([b_copied2, c_copied1])]
        conclusions = [necessary_logic_functions["BiggerOrEqual"].execute_lf([a_copied2, c_copied2])]
        return {"Assumptions": assumptions, "Conclusions": conclusions}

    def infer_operands(self, entity_list):
        # TODO
        return NotImplementedError

    def transform_gt(self, core_gt, entities):
        right_entities = [entity for entity in entities if entity.name == core_gt.operands[1].name and
                          entity.parent_index == 0 and entity.root.logic_function.name == "BiggerOrEqual" and
                          entity is entity.root.operands[0]]
        if len(right_entities) == 0:
            return self.extend_core_gt(core_gt, entities, False)

        right_entity = random.choice(right_entities)
        return {
            "action": True,
            "makeup": False,
            "operands": [core_gt.operands[0], core_gt.operands[1], right_entity.root.operands[1]]
        }

    def extend_core_gt(self, core_gt, entities, transform_gt):
        if core_gt.logic_function.name == "BiggerOrEqual" and core_gt.operands[0] == core_gt.operands[1]:
            return {
                "action": False
            }
        if transform_gt:
            return self.transform_gt(core_gt, entities)
        else:
            return {
                "action": True,
                "makeup": True,
                "makeup_config": [{
                    "requirement_type": "BiggerOrEqual",
                    "a": random.choice(entities),
                    "b": core_gt.operands[0]
                }],
                "operand_retrieval":
                    lambda makeup_conclusions: [makeup_conclusions[0].operands[0]] + core_gt.operands,
            }


class EquivalenceTransitivity(MetaAxiom):
    def __init__(self):
        equivalence_theorem = False
        input_no = 3
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        assumption_types = [assump.logic_function.name for assump in result["Assumptions"]]
        super(EquivalenceTransitivity, self).__init__(input_no=input_no,
                                                      assumption_size=assumption_size,
                                                      conclusion_size=conclusion_size,
                                                      equivalence_theorem=equivalence_theorem,
                                                      assumption_types=assumption_types)

    def execute_th(self, operands, mode="generate"):
        """
        If a=b, b=c, then a=c.
        :param operands: 3 input [a, b, c]
        :return: dict(Assumptions, Conclusions)
        """
        a, b, c = operands
        a_copied_1, a_copied_2, b_copied_1, b_copied_2, c_copied_1, c_copied_2 = \
            a, a, b, b, c, c
        assumptions = [
            necessary_logic_functions["Equivalent"].execute_lf([a_copied_1, b_copied_1]),
            necessary_logic_functions["Equivalent"].execute_lf([b_copied_2, c_copied_1]),
        ]
        conclusions = [
            necessary_logic_functions["Equivalent"].execute_lf([a_copied_2, c_copied_2]),
        ]
        return {
            "Assumptions": assumptions,
            "Conclusions": conclusions,
        }

    def infer_operands(self, entity_list):
        all_equalities = set([entity.root for entity in entity_list
                              if (entity.root.logic_function.name == "Equivalent"
                                  and entity.root.operands[0].name != entity.root.operands[1].name)])
        id_equality = {}
        left_names = {}
        right_names = {}
        for equality in all_equalities:
            current_id = len(id_equality)
            id_equality[current_id] = equality
            if equality.operands[0].name in left_names.keys():
                left_names[equality.operands[0].name].append(current_id)
            else:
                left_names[equality.operands[0].name] = [current_id]
            if equality.operands[1].name in right_names.keys():
                right_names[equality.operands[1].name].append(current_id)
            else:
                right_names[equality.operands[1].name] = [current_id]

        possible_operands = []
        for name in right_names:
            if name in left_names:
                b = id_equality[right_names[name][0]].operands[1]
                for r_equal_id in right_names[name]:
                    a = id_equality[r_equal_id].operands[0]
                    for l_equal_id in left_names[name]:
                        c = id_equality[l_equal_id].operands[1]
                        possible_operands.append([a, b, c])

        if len(possible_operands) == 0:
            return None
        return possible_operands

    @staticmethod
    def assumptions2operands(made_up_conditions):
        assert len(made_up_conditions) == 2
        operands = made_up_conditions[0].operands + [made_up_conditions[1].operands[1]]
        return operands

    def extend_core_gt(self, core_gt, entities, transform_gt):
        if transform_gt:
            # Can't do transform gt with equivalence transitivity
            pass
        if core_gt.logic_function.name == "Equivalent" and core_gt.operands[0].name == core_gt.operands[1].name:
            return {
                "action": True,
                "makeup": True,
                "makeup_config": [
                    {
                        "requirement_type": "Equivalent",
                        "a": random.choice(entities),
                        "b": core_gt.operands[0]
                    },
                    {
                        "requirement_type": "Equivalent",
                        "a": core_gt.operands[1],
                        "b": random.choice(entities),
                        "new_iv": False
                    }
                ],
                "operand_retrieval":
                    lambda makeup_conclusions: [
                        makeup_conclusions[0].operands[0],
                        makeup_conclusions[0].operands[1],
                        makeup_conclusions[1].operands[1]
                    ]
            }
        else:
            return {
                "action": True,
                "makeup": True,
                "makeup_config": [{
                    "requirement_type": "Equivalent",
                    "a": random.choice(entities),
                    "b": core_gt.operands[0]
                }],
                "operand_retrieval":
                    lambda makeup_conclusions: [
                        makeup_conclusions[0].operands[0],
                        core_gt.operands[0],
                        core_gt.operands[1]]
            }


class EquivalenceReflexibility(MetaAxiom):
    def __init__(self):
        equivalence_theorem = False
        input_no = 1
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        assumption_types = [assump.logic_function.name for assump in result["Assumptions"]]
        super(EquivalenceReflexibility, self).__init__(input_no=input_no,
                                                       assumption_size=assumption_size,
                                                       conclusion_size=conclusion_size,
                                                       equivalence_theorem=equivalence_theorem,
                                                       assumption_types=assumption_types)

    def execute_th(self, operands, mode="generate"):
        """
        a=a.
        :param operands: 1 input [a]
        :return: dict(Assumptions, Conclusions)
        """
        a, = operands
        a_copied_1 = a
        a_copied_2 = a
        assumptions = list()
        conclusions = [necessary_logic_functions["Equivalent"].execute_lf([a_copied_1, a_copied_2])]
        return {
            "Assumptions": assumptions,
            "Conclusions": conclusions,
        }

    def infer_operands(self, entity_list):
        return [[entity] for entity in entity_list]

    def extend_core_gt(self, core_gt, entities, transform_gt):
        return {
            "action": False
        }


class EquivalenceSymmetry(MetaAxiom):
    def __init__(self):
        equivalence_theorem = True
        input_no = 2
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        assumption_types = [assump.logic_function.name for assump in result["Assumptions"]]
        super(EquivalenceSymmetry, self).__init__(input_no=input_no,
                                                  assumption_size=assumption_size,
                                                  conclusion_size=conclusion_size,
                                                  equivalence_theorem=equivalence_theorem,
                                                  assumption_types=assumption_types)

    def execute_th(self, operands, mode="generate"):
        """
        If a=b, then b=a.
        :param operands: 2 inputs [a, b]
        :return: dict(Assumptions, Conclusions)
        """
        a, b = operands
        a_copied_1, a_copied_2, b_copied_1, b_copied_2 = a, a, b, b
        assumptions = [
            necessary_logic_functions["Equivalent"].execute_lf([a_copied_1, b_copied_1])
        ]
        conclusions = [
            necessary_logic_functions["Equivalent"].execute_lf([b_copied_2, a_copied_2])
        ]
        return {
            "Assumptions": assumptions,
            "Conclusions": conclusions
        }

    def infer_operands(self, entity_list):
        operands = list()
        for entity in entity_list:
            if entity.root.logic_function.name == "Equivalent":
                operands.append(entity.root.operands)
        if len(operands) == 0:
            return None
        return operands

    @staticmethod
    def assumptions2operands(made_up_conditions):
        assert len(made_up_conditions) == 1
        return made_up_conditions[0].operands

    def extend_core_gt(self, core_gt, entities, transform_gt):
        if transform_gt:
            # Can't do transform gt with equivalence symmetry
            pass
        if core_gt.logic_function.name == "Equivalent" and core_gt.operands[0].name == core_gt.operands[1].name:
            return {
                "action": True,
                "makeup": True,
                "makeup_config": [
                    {
                        "requirement_type": "Equivalent",
                        "a": core_gt.operands[0],
                        "b": random.choice(entities),
                    }
                ],
                "operand_retrieval":
                    lambda makeup_conclusions:
                    makeup_conclusions[0].operands
            }
        else:
            return {
                "action": True,
                "makeup": False,
                "operands": core_gt.operands
            }


class MultiplicationLeftZero(MetaAxiom):
    def __init__(self):
        equivalence_theorem = False
        input_no = 1
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        assumption_types = ["Equivalent"]
        super(MultiplicationLeftZero, self).__init__(input_no=input_no,
                                                     assumption_size=assumption_size,
                                                     conclusion_size=conclusion_size,
                                                     equivalence_theorem=equivalence_theorem,
                                                     assumption_types=assumption_types)

    def execute_th(self, operands, mode="generate"):
        """
        0 * a = 0
        :param operands: 1 inputs [a]
        :return: dict(Assumptions, Conclusions)
        """
        a, = operands
        zero = Entity(name="0", is_constant=True)
        zero_times_a = necessary_numerical_functions["mul"].execute_nf([zero, a])
        assumptions = list()
        conclusions = [necessary_logic_functions["Equivalent"].execute_lf([zero_times_a, zero])]
        return {
            "Assumptions": assumptions,
            "Conclusions": conclusions
        }

    def infer_operands(self, entity_list):
        raise NotImplementedError

    def transform_gt(self, core_gt, entities):
        all_operands = search_operator_operands_in_gt(core_gt, "mul")
        suitable_operands = [(operands[1],) for operands in all_operands if operands[0].name == '0']
        if len(suitable_operands) == 0:
            return self.extend_core_gt(core_gt, entities, False)

        operands = random.choice(suitable_operands)
        return {
            "action": True,
            "makeup": False,
            "operands": operands,
            "substitution_retrieval":
                lambda makeup_conclusion, proof_conclusion:
                [core_gt.ent_dic[operands[0].parent_index], proof_conclusion.operands[1]]
        }

    def extend_core_gt(self, core_gt, entities, transform_gt):
        if transform_gt:
            return self.transform_gt(core_gt, entities)
        return {
            "action": False,
        }


class MultiplicationRightZero(MetaAxiom):
    def __init__(self):
        equivalence_theorem = False
        input_no = 1
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        assumption_types = ["Equivalent"]
        super(MultiplicationRightZero, self).__init__(input_no=input_no,
                                                      assumption_size=assumption_size,
                                                      conclusion_size=conclusion_size,
                                                      equivalence_theorem=equivalence_theorem,
                                                      assumption_types=assumption_types)

    def execute_th(self, operands, mode="generate"):
        """
        a * 0 = 0
        :param operands: 1 inputs [a]
        :return: dict(Assumptions, Conclusions)
        """
        a, = operands
        zero = Entity(name="0", is_constant=True)
        a_times_zero = necessary_numerical_functions["mul"].execute_nf([a, zero])
        assumptions = list()
        conclusions = [necessary_logic_functions["Equivalent"].execute_lf([a_times_zero, zero])]
        return {
            "Assumptions": assumptions,
            "Conclusions": conclusions
        }

    def infer_operands(self, entity_list):
        raise NotImplementedError

    def transform_gt(self, core_gt, entities):
        all_operands = search_operator_operands_in_gt(core_gt, "mul")
        suitable_operands = [(operands[0],) for operands in all_operands if operands[1].name == '0']
        if len(suitable_operands) == 0:
            return self.extend_core_gt(core_gt, entities, False)

        operands = random.choice(suitable_operands)
        return {
            "action": True,
            "makeup": False,
            "operands": operands,
            "substitution_retrieval":
                lambda makeup_conclusion, proof_conclusion:
                [core_gt.ent_dic[operands[0].parent_index], proof_conclusion.operands[1]]
        }

    def extend_core_gt(self, core_gt, entities, transform_gt):
        if transform_gt:
            return self.transform_gt(core_gt, entities)
        return {
            "action": False,
        }


class AdditionMultiplicationDistribution(MetaAxiom):
    def __init__(self):
        equivalence_theorem = False
        input_no = 3
        test_entities = [Entity(name="{}".format(i)) for i in range(input_no)]
        result = self.execute_th(test_entities)
        assumption_size, conclusion_size = len(result["Assumptions"]), len(result["Conclusions"])
        assumption_types = [assump.logic_function.name for assump in result["Assumptions"]]
        super(AdditionMultiplicationDistribution, self).__init__(input_no=input_no,
                                                                 assumption_size=assumption_size,
                                                                 conclusion_size=conclusion_size,
                                                                 equivalence_theorem=equivalence_theorem,
                                                                 assumption_types=assumption_types)

    def execute_th(self, operands, mode="generate"):
        """
        This shall be treated as a normal arithmetic manipulation axiom, it shouldn't have to do with logic statements.
        :param operands:
        :return:
        """
        a, c, d = operands
        a_copy1, a_copy2, a_copy3, a_copy4, a_copy5, a_copy6 = \
            a, a, a, a, a, a
        c_copy1, c_copy2, c_copy3, c_copy4 = \
            c, c, c, c
        d_copy1, d_copy2, d_copy3, d_copy4 = \
            d, d, d, d
        # Construct the first conclusion
        c_and_d1 = necessary_numerical_functions["add"].execute_nf([c_copy1, d_copy1])
        lhs1 = necessary_numerical_functions["mul"].execute_nf([a_copy1, c_and_d1])
        a_times_c1 = necessary_numerical_functions["mul"].execute_nf([a_copy2, c_copy2])
        a_times_d1 = necessary_numerical_functions["mul"].execute_nf([a_copy3, d_copy2])
        rhs1 = necessary_numerical_functions["add"].execute_nf([a_times_c1, a_times_d1])
        con1 = necessary_logic_functions["Equivalent"].execute_lf([lhs1, rhs1])

        c_and_d2 = necessary_numerical_functions["add"].execute_nf([c_copy3, d_copy3])
        lhs2 = necessary_numerical_functions["mul"].execute_nf([c_and_d2, a_copy4])
        a_times_c2 = necessary_numerical_functions["mul"].execute_nf([a_copy5, c_copy4])
        a_times_d2 = necessary_numerical_functions["mul"].execute_nf([a_copy6, d_copy4])
        rhs2 = necessary_numerical_functions["add"].execute_nf([a_times_c2, a_times_d2])
        con2 = necessary_logic_functions["Equivalent"].execute_lf([lhs2, rhs2])

        return {
            "Assumptions": [],
            "Conclusions": [con1, con2]
        }

        # """
        # a * b(b=c+d) or b(b=c+d) * a = a * c + a * d
        # :param operands: 3 inputs [a, c, d]
        # :return: dict(Assumptions, Conclusions)
        # """
        # a, c, d = operands
        # gt = a.root
        #
        # # First check: a, c and d are in the same gt
        # if (c.root is not d.root) or (a.root is not c.root) or (gt is None):
        #     return {
        #         "Assumptions": list(),
        #         "Conclusions": list()
        #     }
        #
        # # Determine whether the form of the existing entity is lhs or rhs
        # lhs, rhs = False, False
        # if c.parent_index == d.parent_index \
        #         and isinstance(gt.ent_dic[c.parent_index], Entity) \
        #         and gt.ent_dic[c.parent_index].recent_numerical_function.name == "add":
        #     # We have c+d
        #     # Might be lhs
        #     b = gt.ent_dic[c.parent_index]
        #     if b.parent_index == a.parent_index \
        #             and isinstance(gt.ent_dic[b.parent_index], Entity) \
        #             and gt.ent_dic[a.parent_index].recent_numerical_function.name == "mul":
        #         # We have a * (c + d) or (c + d) * a
        #         lhs = True
        # else:
        #     # Might be rhs
        #     if (a.parent_index == c.parent_index or a.parent_index == d.parent_index) \
        #             and isinstance(gt.ent_dic[a.parent_index], Entity) \
        #             and gt.ent_dic[a.parent_index].recent_numerical_function.name == "mul":
        #         # We have a * c or d
        #         g_parent = gt.ent_dic[gt.ent_dic[a.parent_index].parent_index]
        #         if isinstance(g_parent, Entity) \
        #                 and g_parent.recent_numerical_function.name == "add":
        #             # We have a * c or d + ...
        #             l_operands, r_operands = g_parent.operands
        #             if l_operands.operands is not None and r_operands.operands is not None:
        #                 l_operand_names = [l_operand.name for l_operand in l_operands.operands]
        #                 r_operand_names = [r_operand.name for r_operand in r_operands.operands]
        #                 all_operand_names = sorted(l_operand_names + r_operand_names)
        #                 supposed_operand_names = sorted([a.name, a.name, c.name, d.name])
        #                 if all_operand_names == supposed_operand_names:
        #                     rhs = True
        #
        # assumptions = list()
        # if lhs and not rhs:
        #     parent_node = g.ent_dic[a.parent_index])
        #     # parent node is a * (c + d)
        #     a_copied1, a_copied2 = a, a
        #     c_copied, d_copied = c, d
        #     a_mul_c = necessary_numerical_functions["mul"].execute_nf([a_copied1, c_copied])
        #     a_mul_d = necessary_numerical_functions["mul"].execute_nf([a_copied2, d_copied])
        #     rhs = necessary_numerical_functions["add"].execute_nf([a_mul_c, a_mul_d])
        #     expansion = necessary_logic_functions["Equivalent"].execute_lf([parent_node, rhs])
        #     conclusions = [expansion]
        # elif rhs and not lhs:
        #     g_parent_node = g.ent_dic[gt.ent_dic[a.parent_index].parent_index])
        #     # grand parent node is a * c + a * d
        #     a_copied, c_copied, d_copied = a, c, d
        #     b = necessary_numerical_functions["add"].execute_nf([c_copied, d_copied])
        #     a_mul_b = necessary_numerical_functions["mul"].execute_nf([a, b])
        #     expansion = necessary_logic_functions["Equivalent"].execute_lf([g_parent_node, a_mul_b])
        #     conclusions = [expansion]
        # elif not lhs and not rhs:
        #     conclusions = list()
        # else:
        #     raise NotImplementedError
        #
        # return {
        #     "Assumptions": assumptions,
        #     "Conclusions": conclusions
        # }

    def infer_operands(self, entity_list):
        possible_operands = []
        for entity in entity_list:
            if entity.recent_numerical_function is not None and entity.recent_numerical_function.name == "mul":
                a, b = entity.operands
                if a.recent_numerical_function is not None and a.recent_numerical_function.name == "add":
                    a_addition = True
                else:
                    a_addition = False
                if b.recent_numerical_function is not None and b.recent_numerical_function.name == "add":
                    b_addition = True
                else:
                    b_addition = False

                if a_addition:
                    possible_operands.append([b] + a.operands)
                if b_addition:
                    possible_operands.append([a] + b.operands)

        if len(possible_operands) == 0:
            return None
        return possible_operands

    def extend_core_gt(self, core_gt, entities, transform_gt):
        if core_gt.logic_function.name == "Equivalent":
            a_original = core_gt.operands[0]
            c_original, d_original = random.choices(entities, k=2)
            return {
                "action": True,
                "makeup": False,
                "operands": [a_original, c_original, d_original],
                "substitution_retrieval":
                    lambda makeup_conclusion, proof_conclusion:
                    [proof_conclusion.operands[0].operands[0], core_gt.operands[1]]
            }
        elif core_gt.logic_function.name == "BiggerOrEqual":
            return {
                "action": False
            }
        else:
            raise NotImplementedError
