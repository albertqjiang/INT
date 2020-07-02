import random
import re
from pprint import pprint
from copy import deepcopy
from logic.utils import standard_logic_functions, standard_numerical_functions
from logic.logic import Entity, Proof
from legacy.pseq.errors import InputError


def replace(string, substitutions):
    substrings = sorted(substitutions, key=len, reverse=True)
    regex = re.compile('|'.join(map(re.escape, substrings)))
    return regex.sub(lambda match: substitutions[match.group(0)], string)


def create_theorem(name, assumption_logic_statements, objective_logic_statement, max_input_no=9):
    input_mapping = dict()
    for input_no in range(1, 1 + max_input_no):
        input_in_ls = [("input{}".format(input_no) in assump.name) for assump in assumption_logic_statements]
        input_in_ls.append("input{}".format(input_no) in objective_logic_statement.name)
        if True in input_in_ls:
            input_mapping["input{}".format(input_no)] = "input{}".format(len(input_mapping) + 1)
    # print(input_mapping)

    objective_logic_function = objective_logic_statement.logic_function
    objective_entities = objective_logic_statement.entities
    constraints = [[replace(entity.name, input_mapping), "input{}".format(i + 1 + len(input_mapping))]
                   for i, entity in enumerate(objective_entities)]
    # print(constraints)

    assumptions = list()
    for als in assumption_logic_statements:
        alf = als.logic_function
        ales = als.entities
        a_substituted_entities = [replace(entity.name, input_mapping) for entity in ales]
        a_substituted_indices = [int(se.lstrip("input")) - 1 for se in a_substituted_entities]
        assumptions.append([alf, a_substituted_indices])
    # print(assumptions)

    conclusions = [[objective_logic_function, list()]]
    c_substituted_entities = [replace(entity.name, input_mapping) for entity in objective_entities]
    for cse in c_substituted_entities:
        for cons in constraints:
            if cons[0] == cse:
                index = int(cons[1].lstrip("input")) - 1
                conclusions[0][1].append(index)
                break
    # print(conclusions)

    return StringTheorem(name=name, input_no=len(input_mapping) + len(objective_entities),
                         input_constraints=constraints, assumptions=assumptions, conclusions=conclusions)


class StringTheorem:
    def __init__(self, name, input_no, input_constraints, assumptions, conclusions):
        self.name = name
        self.input_no = input_no
        # Input constraints should be a sequence of equal expressions
        self.input_constraints = input_constraints
        self.assumptions = assumptions
        self.conclusions = conclusions

    def execute(self, inputs):
        if len(inputs) != self.input_no:
            raise InputError("Inputs have length {} while it should have length {}.".format(len(inputs), self.input_no),
                             "Input length mismatch.")
        elif not self.input_valid(inputs):
            raise InputError("Inputs {} don't satisfy the conditions for the theorem {}.".format(
                [inp.to_string() for inp in inputs], self.name), "Inputs not valid.")
        else:
            left = list()
            for assump in self.assumptions:
                left.append(assump[0].execute_lf([inputs[i] for i in assump[1]]))
            right = list()
            for conclu in self.conclusions:
                right.append(conclu[0].execute_lf([inputs[i] for i in conclu[1]]))
            return {"Assumptions": set(left), "Conclusions": set(right)}

    def input_valid(self, inputs, numerical_evaluator=None):
        if not numerical_evaluator:
            from logic.utils import NumericalEvaluator
            numerical_evaluator = NumericalEvaluator()
        for constraint in self.input_constraints:
            customized_constraint = self._customize_constraint(constraint, inputs)
            if not numerical_evaluator.equal_string_pair((customized_constraint[0], customized_constraint[1])):
                return False
        return True

    def to_string(self):
        raise NotImplementedError

    def _customize_constraint(self, constraint, inputs):
        input_mapping = {"input{}".format(1 + i): inputs[i].name for i in range(self.input_no)}
        customized_constraint = deepcopy(constraint)
        # print(customized_constraint)
        customized_constraint[0] = replace(constraint[0], input_mapping)
        customized_constraint[1] = replace(constraint[1], input_mapping)
        # print(customized_constraint)
        return customized_constraint


def random_search_theorem():
    modified_numerical_functions = standard_numerical_functions
    del modified_numerical_functions["identity"]
    del modified_numerical_functions["geometric_mean"]
    x = Entity(name="input1")
    y = Entity(name="input2")
    z = Entity(name="input3")
    x_sub_y = standard_numerical_functions["sub"].execute_lf([x, y])
    x_and_y = standard_numerical_functions["add"].execute_lf([x, y])
    x_sub_y_plus_y = standard_numerical_functions["add"].execute_lf([x_sub_y, y])
    x_sub_y_plus_y_sqr = standard_numerical_functions["sqr"].execute_lf([x_sub_y_plus_y])
    x_sqr = standard_numerical_functions["sqr"].execute_lf([x])
    x_real = standard_logic_functions["Real"].execute_lf([x])
    y_real = standard_logic_functions["Real"].execute_lf([y])
    x_sqr_non_negative = standard_logic_functions["NonNegative"].execute_lf([x_sqr])

    real_square_nn = StringTheorem(name="Real Square Non-negative", input_no=2,
                                   input_constraints=[[x_sqr.name, y.name]],
                                   assumptions=((standard_logic_functions["Real"], (0,)),),
                                   conclusions=((standard_logic_functions["NonNegative"], (1,)),))
    everything_is_real = StringTheorem(
        name="Everything is real", input_no=1,
        input_constraints=[], assumptions=[], conclusions=[[standard_logic_functions["Real"], [0]]]
    )
    first_principle_of_inequality = StringTheorem(
        name="First principle of inequality", input_no=5,
        input_constraints=[["input1 + input3", "input4"], ["input2 + input3", "input5"]],
        assumptions=[[standard_logic_functions["BiggerOrEqual"], [0, 1]]],
        conclusions=[[standard_logic_functions["BiggerOrEqual"], [3, 4]]]
    )
    # print(real_square_nn.execute_lf())
    # print(real_square_nn.input_valid([x, x_sub_y_plus_y]))
    proof = Proof(entities=[x, y], axioms=[real_square_nn, everything_is_real, first_principle_of_inequality],
                  assumptions=[x_real, y_real], objectives=[x_sqr_non_negative])
    pprint([gt.name for gt in proof.ground_truth])

    for _ in range(20000):
        ax = random.choice(proof.lemmas)
        chosen_entities = random.choices(proof.entities, k=ax.input_no)
        try:
            execution = ax.execute(chosen_entities)
            assump, conclu = execution["Assumptions"], execution["Conclusions"]
            if proof.statements_all_valid(assump):
                for con in conclu:
                    if not proof.statements_all_valid([con]):
                        proof.ground_truth.append(con)
        except InputError:
            pass
        if len(proof.entities) < 20:
            nf = random.choices(list(modified_numerical_functions.values()), k=1)[0]
            proof.entities.append(nf.execute_nf(random.choices(proof.entities, k=nf.input_no)))

    print([ent.name for ent in proof.entities])
    pprint([gt.name for gt in proof.ground_truth])


if __name__ == "__main__":
    x = Entity(name="input2")
    x_real = standard_logic_functions["Real"].execute_lf([x])
    x_sqr = standard_numerical_functions["sqr"].execute_nf([x])
    x_sqr_nn = standard_logic_functions["NonNegative"].execute_lf([x_sqr])

    name = "test"
    assumptions = [x_real]
    objective = x_sqr_nn

    theorem = create_theorem(name=name, assumption_logic_statements=assumptions, objective_logic_statement=objective)

    theorem.execute([x, x_sqr])
    # random_search_theorem()
