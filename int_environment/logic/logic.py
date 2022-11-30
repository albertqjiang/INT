import string
import math
import pickle
from copy import deepcopy
from abc import ABC, abstractmethod


def _copy_input(inputs):
    copy_inputs = []
    for inp in inputs:
        copy_inputs.append(deepcopy(inp))
    return copy_inputs


class LogicFunction:
    def __init__(self, name, input_no):
        self.name = name
        self.input_no = input_no

    def execute_lf(self, inputs):
        if self.input_no:
            if len(inputs) != self.input_no:
                raise AssertionError("Required {} inputs but got {}.".format(self.input_no, len(inputs)),
                                     "Input number mismatch.")
        new_degree = 1
        inputs = _copy_input(inputs)
        return LogicStatement(
            logic_function=self,
            operands=inputs, degree=new_degree
        )

    def to_string(self):
        return self.name


class NumericalFunction:
    def __init__(self, name, input_no=None):
        self.name = name
        self.input_no = input_no

    def execute_nf(self, inputs):
        if self.input_no:
            if len(inputs) != self.input_no:
                raise AssertionError("Required {} inputs but got {}.".format(self.input_no, len(inputs)),
                                     "Input number mismatch.")
        new_degree = sum(operand.degree for operand in set(inputs)) + 1
        # make unique leaf node (ground entities)
        inputs = _copy_input(inputs)
        return Entity(recent_numerical_function=self, operands=inputs, degree=new_degree)

    def to_string(self):
        return self.name


class Entity:
    def __init__(self, name=None, recent_numerical_function=None, operands=None, is_iv=False, is_constant=False, degree=0):
        self.degree = degree
        self.index = None
        self.parent_index = None
        self.root = None
        self.recent_numerical_function = recent_numerical_function
        self.operands = operands
        if name is not None:
            assert operands is None
            assert recent_numerical_function is None
            self.name = name
        else:
            self.update_name()
        self.is_constant = is_constant
        self.iv = is_iv

    def update_name(self):
        self.name = (self.recent_numerical_function.name +
                     " ( " + " , ".join([inp.to_string() for inp in self.operands]) + " )")

    def to_string(self):
        return self.name


class LogicStatement:
    def __init__(self, logic_function, operands, degree=1, premise=None):
        self.logic_function = logic_function
        self.operands = operands
        self.degree = degree
        self.premise = premise
        self.indexing()
        self.update_name()

    def indexing(self):
        node_count = []
        self.ent_dic = {0: self}
        self.ent = []
        def _graph_index(entity, parent_index):
            if entity.operands is None:
                assert entity not in self.ent
                node_count.append(1)
                entity.index = len(node_count)
                entity.parent_index = parent_index
                entity.root = self
                self.ent_dic[entity.index] = entity
                self.ent.append(entity)
            else:
                assert entity not in self.ent
                node_count.append(1)
                entity.index = len(node_count)
                entity.parent_index = parent_index
                entity.root = self
                self.ent_dic[entity.index] = entity
                self.ent.append(entity)
                for ent in entity.operands:
                    _graph_index(ent, entity.index)
        for ent in self.operands:
            _graph_index(ent, 0)

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

    def to_string(self):
        return self.name


class Theorem:
    theorem_count = 0

    def __init__(self, name, input_no, input_constraints, assumptions, conclusions):
        """
        The examples are made for the case P(x), Q(x, y) => R(s(x, y))
        :param input_no: the number of inputs to the theorem, 3 (x, y, s(x, y))
        :param input_constraints: the constraints on the inputs, [(s, (0, 1), (2,))]
        :param assumptions: the assumptions of the theorem, [(P, (0, )), (Q, (0, 1))]
        :param conclusions: the conclusions of the theorem, [(R, (0, 1))]
        """
        self.name = name
        self.input_no = input_no
        self.input_constraints = input_constraints
        self.assumptions = assumptions
        self.conclusions = conclusions
        Theorem.theorem_count += 1

    def execute(self, inputs):
        if len(inputs) != self.input_no:
            raise AssertionError(
                "Inputs have length {} while it should have length {}.".format(len(inputs), self.input_no),
                "Input length mismatch.")
        elif not self.input_valid(inputs):
            raise AssertionError("Inputs {} don't satisfy the conditions for the theorem {}.".format(
                [inp.to_string() for inp in inputs], self.name), "Inputs not valid.")
        else:
            left = []
            for assump in self.assumptions:
                left.append(assump[0].execute_lf([inputs[i] for i in assump[1]]))
            right = []
            for conclu in self.conclusions:
                right.append(conclu[0].execute_lf([inputs[i] for i in conclu[1]]))
            return set(left), set(right)

    def input_valid(self, inputs, numerical_evaluator=None):
        if not numerical_evaluator:
            from int_environment.logic.utils import NumericalEvaluator
            numerical_evaluator = NumericalEvaluator()
        for constraint in self.input_constraints:
            if not numerical_evaluator.equal_pair(
                    (constraint[0].execute_nf(inputs=[inputs[ind] for ind in constraint[1]]), inputs[constraint[2][0]])
            ):
                return False
        return True

    def numerical_input_valid(self, numerical_evaluator, inputs):
        raise NotImplementedError

    def to_string(self, input_strings=None):
        """

        :return: the human-readable representation of a theorem in string
        """
        if not input_strings:
            input_strings = list(string.ascii_lowercase)[:self.input_no]
        input_symbols = [Entity(input_strings[i]) for i in range(self.input_no)]

        theorem_rep = "Theorem name: {}\n".format(self.name)
        theorem_rep += "For all " + ", ".join(input_strings) + ",\nif "

        for constraint in self.input_constraints:
            desired_condition_output = constraint[0].execute_nf(
                inputs=[input_symbols[ind] for ind in constraint[1]]).to_string()
            real_condition_output = input_strings[constraint[2][0]]
            theorem_rep += real_condition_output + " = " + desired_condition_output + ", "

        for assumption in self.assumptions:
            theorem_rep += assumption[0].execute_lf(inputs=[input_symbols[ind] for ind in assumption[1]]).name + ", "

        theorem_rep += "\nthen "
        for conclusion in self.conclusions:
            theorem_rep += conclusion[0].execute_lf(inputs=[input_symbols[ind] for ind in conclusion[1]]).name
        theorem_rep += " hold.\n\n"

        return theorem_rep


class Proof:
    def __init__(self, axioms, conditions, objectives):
        self.assumptions = conditions
        self.ground_truth = [assu for assu in conditions]
        self.axioms = axioms
        # lemmas are extendable
        self.lemmas = [ax for ax in axioms]
        self.objectives = objectives
        self.proved = False
        self.logic_chain = "The proof of the theorem is shown as follows:\n"
        # print(self.print_proof_status())

    def get_observation(self):
        raw_observation = dict()
        raw_observation["ground_truth"] = self.ground_truth
        raw_observation["lemmas"] = self.lemmas
        raw_observation["objectives"] = self.objectives
        return raw_observation

    def apply_theorem(self, theorem, operands):
        results = theorem.execute_th(operands)
        assumptions, conclusions = \
            results["Assumptions"], results["Conclusions"]

        if self.statements_all_valid(assumptions):
            if self.statements_all_valid(conclusions):
                return "REWARD_DUPLICATED_RESULTS"
            else:
                # Assign the assumptions as premises to conclusions
                original_assumptions = [self.find_in_ground_truth(ass) for ass in assumptions]
                for con in conclusions:
                    con.premise = original_assumptions
                self.ground_truth.extend(conclusions)
        else:
            return "REWARD_ASSUMPTION_INVALID"

        self.logic_chain += theorem.name + "\n inputs:\n" + " ".join([operand.name for operand in operands])

        for ls in assumptions + conclusions:
            ls.indexing()
        if self.statements_all_valid(self.objectives):
            self.proved = True
            return "REWARD_PROOF_COMPLETE"
        else:
            return "REWARD_THEOREM_PROCEEDED"

    def apply_theorem_get_reward(self, theorem, operands, reward_scheme, reward_scaling_temp=0):
        """
        An upgraded version of apply theorem.
        :param theorem:
        :param operands:
        :param reward_scheme:
        :param reward_scaling_temp:
        :return: reward * exp(len(sequence) * reward_scaling_temp)
        """
        if len(operands) == theorem.input_no:
            try:
                reward_string = self.apply_theorem(theorem=theorem, operands=operands)
            except AssertionError:
                reward_string = "REWARD_INPUT_INVALID"
        else:
            reward_string = "REWARD_NULL"
        reward = reward_scheme[reward_string]
        if reward > 0:
            reward = reward * math.exp(reward_scaling_temp * theorem.input_no)
        return reward, reward_string

    def apply_theorem_get_conclusions(self, theorem, operands):
        """
        Modified version of apply_theorem for HER.
        :param theorem:
        :param operands:
        :return:
        """
        results = theorem.execute_th(operands)
        assumptions, conclusions = \
            results["Assumptions"], results["Conclusions"]

        if self.statements_all_valid(assumptions):
            if self.statements_all_valid(conclusions):
                return "REWARD_DUPLICATED_RESULTS", conclusions
            else:
                # Assign the assumptions as premises to conclusions
                original_assumptions = [self.find_in_ground_truth(ass) for ass in assumptions]
                for con in conclusions:
                    con.premise = original_assumptions
                self.ground_truth.extend(conclusions)
        else:
            return "REWARD_ASSUMPTION_INVALID", conclusions

        self.logic_chain += theorem.name + "\n inputs:\n" + " ".join([operand.name for operand in operands])

        if self.statements_all_valid(self.objectives):
            self.proved = True
            return "REWARD_PROOF_COMPLETE", conclusions
        else:
            return "REWARD_THEOREM_PROCEEDED", conclusions

    def apply_theorem_get_conclusions_and_reward(self, theorem, operands, reward_scheme, reward_scaling_temp=0):
        """
        Modified version of apply_theorem_get_reward for HER.
        :param theorem:
        :param operands:
        :param reward_scheme:
        :param reward_scaling_temp:
        :return:
        """
        if len(operands) == theorem.input_no:
            try:
                reward_string, conclusions = self.apply_theorem_get_conclusions(theorem=theorem, operands=operands)
            except AssertionError:
                reward_string = "REWARD_INPUT_INVALID"
                conclusions = None
        else:
            reward_string = "REWARD_NULL"
            conclusions = None
        reward = reward_scheme[reward_string]
        if reward > 0:
            reward = reward * math.exp(reward_scaling_temp * theorem.input_no)
        return reward, reward_string, conclusions

    def statements_all_valid(self, statements):
        ground_truth_strings = [gs.name for gs in self.ground_truth]
        for stn in statements:
            if stn.name not in ground_truth_strings:
                return False
        return True

    def find_in_ground_truth(self, statement):
        for gs in self.ground_truth:
            if statement.name == gs.name:
                return gs
        return False

    def trim_ground_truth(self, max_number_of_ground_truth):
        if len(self.ground_truth) > max_number_of_ground_truth:
            self.ground_truth = self.assumptions + \
                                self.ground_truth[-(max_number_of_ground_truth - len(self.assumptions)):]

    def print_proof_status(self):
        proof_status = ""
        proof_status += "*" * 200 + "\n"
        proof_status += "Axioms of the proof:\n"
        for ind, axiom in enumerate(self.axioms):
            proof_status += "{}. ".format(ind + 1) + axiom.name
        proof_status += "\nAssumptions of the proof:\n"
        for ind, assumption in enumerate(self.assumptions):
            proof_status += "{}. ".format(ind+1) + assumption.name + ", "
        proof_status += "\nObjectives of the proof:\n"
        for ind, objective in enumerate(self.objectives):
            proof_status += "{}. ".format(ind+1) + objective.name + ", "
        proof_status += "\n Is the proof completed? {}\n\n".format(self.proved)
        proof_status += self.logic_chain
        proof_status += "*" * 200
        proof_status += "\n"
        return proof_status

    def is_proved(self):
        return self.proved


class Agent(ABC):
    @abstractmethod
    def step(self, proof):
        raise NotImplementedError
