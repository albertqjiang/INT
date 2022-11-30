from copy import deepcopy

from proof_system.numerical_functions import necessary_numerical_functions
from logic.logic import Entity, LogicStatement

import random
random.seed(0)


class EmptyLogicStatement:
    def __init__(self, logic_function, operands, degree=1, premise=None):
        self.logic_function = logic_function
        self.operands = operands
        self.degree = degree
        self.premise = premise
        if logic_function is not None:
            self.update_name()

    def update_name(self):
        def _update_name(entity):
            if entity.operands is not None:
                for ent in entity.operands:
                    _update_name(ent)
                entity.update_name()

        for ent in self.operands:
            _update_name(ent)
        self.name = (self.logic_function.name +
                     " ( " + " , ".join([inp.to_string() for inp in self.operands]) + " )")


def search_operator_operands_in_gt(logic_statement, operator_type):
    assert operator_type in necessary_numerical_functions
    # Take a logic statement and a given operator type, return all operands of operators of the given type
    operands = [tuple(ent.operands) for ent in logic_statement.ent_dic.values()
                if is_entity(ent) and is_structured(ent, operator_type)]
    operands = set(operands)
    return sorted(operands, key=lambda x: x[0].name)


def substitution(x, y):
    # Given x, y, where ls(x) is the logic statement x is in
    # return ls(y)
    ls_x = x.root
    ls_copy = deepcopy(ls_x)
    x_parent_node = ls_copy.ent_dic[x.parent_index]
    for ind, operand in enumerate(x_parent_node.operands):
        if operand.index == x.index:
            replace_ind = ind
            x_parent_node.operands[replace_ind] = deepcopy(y)
        else:
            pass
    ls_copy.indexing()
    ls_copy.update_name()
    return ls_copy


def sub_tree_diff(ls):
    # Find the subtree that differentiates the lhs and rhs of the logic statement
    lhs, rhs, = deepcopy(ls.operands)
    if lhs.name == rhs.name:
        return [None, None]
    while True:
        if lhs.recent_numerical_function is None or rhs.recent_numerical_function is None:
            return [lhs, rhs]
        if lhs.recent_numerical_function.name != rhs.recent_numerical_function.name:
            return [lhs, rhs]

        if lhs.recent_numerical_function.input_no == 1:
            assert lhs.operands[0].name != rhs.operands[0].name
            lhs = deepcopy(lhs.operands[0])
            rhs = deepcopy(rhs.operands[0])
        elif lhs.recent_numerical_function.input_no == 2:
            if lhs.operands[0].name != rhs.operands[0].name:
                if lhs.operands[1].name != rhs.operands[1].name:
                    return [lhs, rhs]
                else:
                    lhs = deepcopy(lhs.operands[0])
                    rhs = deepcopy(rhs.operands[0])
            elif lhs.operands[1].name != rhs.operands[1].name:
                lhs = deepcopy(lhs.operands[1])
                rhs = deepcopy(rhs.operands[1])
            else:
                raise AssertionError


def all_different_subtrees(ls):
    # All different subtrees, in the order of parents to children
    lhs, rhs = sub_tree_diff(ls)
    if lhs is None:
        return [(lhs, rhs)]
    all_diff_subtrees = list()
    while is_entity(lhs):
        all_diff_subtrees.append([lhs, rhs])
        lhs = ls.ent_dic[lhs.parent_index]
        rhs = ls.ent_dic[rhs.parent_index]
    return list(reversed(all_diff_subtrees))


def get_entity_coding_from_ls(ls, entity):
    # Use DPS here
    entity_fronts = []
    for i, ent in enumerate(ls.operands):
        if (len(entity.name) == 1 and entity.name in ent.name.split()) or \
                (len(entity.name) != 1 and entity.name in ent.name):
            entity_fronts.append((ent, [i, ]))
    while len(entity_fronts):
        entity_to_search, coding = entity_fronts.pop()
        if entity_to_search == entity:
            return coding

        if entity_to_search.recent_numerical_function is not None:
            for j, ent in enumerate(entity_to_search.operands):
                if (len(entity.name) == 1 and entity.name in ent.name.split()) or \
                        (len(entity.name) != 1 and entity.name in ent.name):
                    further_coding = coding + [j]
                    entity_fronts.append((ent, further_coding))


def get_entity_from_ls_and_coding(ls, coding):
    lhs, rhs, = ls.operands
    ls_root = lhs.root
    current_ent = deepcopy(ls)
    for code in coding:
        current_ent = deepcopy(current_ent.operands[code])
    return ls_root.ent_dic[current_ent.index]


def search_entity_with_name_in_ls(ls, entity_name):
    all_entities = []
    entity_fronts = [ent for ent in ls.operands]
    while len(entity_fronts) > 0:
        entity_to_search = entity_fronts.pop()
        if entity_to_search.name == entity_name:
            all_entities.append(entity_to_search)

        if entity_to_search.recent_numerical_function is not None:
            entity_fronts.extend(entity_to_search.operands)
    return all_entities


def side_of_an_entity(ent):
    ls = ent.root
    current_index = ent.index
    parent_index = ent.parent_index
    while parent_index != 0:
        current_index = parent_index
        parent_index = ls.ent_dic[current_index].parent_index
    if ls.operands[0].index == current_index:
        return "left"
    elif ls.operands[1].index == current_index:
        return "right"
    else:
        raise NotImplementedError


def numerical_problem_axiom_order_valid(steps):
    if len(steps) % 2 == 0:
        return False
    if steps[0]["lemma"].name == "EquivalenceSubstitution":
        return False
    for i, step in enumerate(steps):
        if i != 0:
            if i % 2 == 0:
                if step["lemma"].name != "EquivalenceSubstitution":
                    return False
            elif i % 2 == 1:
                if step["lemma"].name == "EquivalenceSubstitution":
                    return False
            else:
                raise NotImplementedError
    return True


def general_problem_axiom_order_valid(steps):
    for i in range(len(steps) - 1):
        if steps[i]["lemma"].name == "EquivalenceSubstitution" \
                and steps[i + 1]["lemma"].name == "EquivalenceSubstitution":
            return False
    return True


def is_entity(ent):
    if isinstance(ent, Entity):
        return True
    return False


def is_structured(ent, operator_name):
    if ent.recent_numerical_function is not None and ent.recent_numerical_function.name == operator_name:
        return True
    return False


def is_ls(ls):
    if isinstance(ls, LogicStatement):
        return True
    return False


def is_empty_ls(ls):
    if isinstance(ls, EmptyLogicStatement):
        return True
    return False


def is_ls_type(ls, type_name):
    if isinstance(ls, LogicStatement) and ls.logic_function.name == type_name:
        return True
    return False


def sample_entity(base_entity, entities, terminal_prob=0.5):
    current_entity = deepcopy(base_entity)
    while random.random() > terminal_prob:
        operator = random.choice(necessary_numerical_functions.values())
        if operator.input_no == 1:
            current_entity = deepcopy(operator.execute_nf([current_entity]))
        elif operator.input_no == 2:
            operands = [current_entity, random.choice(entities)]
            random.shuffle(operands)
            current_entity = deepcopy(operator.execute_nf(operands))
        else:
            raise NotImplementedError
    return current_entity
