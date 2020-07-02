import random
from copy import deepcopy
from operator import itemgetter
from collections import OrderedDict

from torch.utils import data as data_handler

from proof_system.prover import Prover
from proof_system.all_axioms import all_axioms_to_prove, generation_type, all_axioms
from proof_system.logic_functions import necessary_logic_functions
from proof_system.numerical_functions import necessary_numerical_functions
from proof_system.utils import is_ls, is_empty_ls, is_entity
from logic.logic import Entity


def generate_starting_ents(degree=0):
    a = Entity("a", is_iv=True)
    b = Entity("b", is_iv=True)
    c = Entity("c", is_iv=True)
    independent_variables = [a, b, c]
    ent_dict = all_entities_to_a_degree(
        atoms=independent_variables,
        operators=necessary_numerical_functions.values(),
        degree=degree
    )
    return independent_variables, ent_dict


def all_entities_to_a_degree(atoms, operators, degree):
    """Generate entities up to a certain degree."""
    binary_operators = [operator for operator in operators if operator.input_no == 2]
    singular_operators = [operator for operator in operators if operator.input_no == 1]
    entities = OrderedDict()
    entities[0] = [atom for atom in atoms]
    for d in range(1, 1 + degree):
        entities[d] = list()

        # Binary operators
        for d1 in range(0, d):
            d2 = d - 1 - d1
            for entity1 in entities[d1]:
                for entity2 in entities[d2]:
                    for operator in binary_operators:
                        copied_entity1, copied_entity2 = entity1, entity2
                        result = operator.execute_nf([copied_entity1, copied_entity2])
                        entities[d].append(result)

        # Singular operators
        for entity in entities[d - 1]:
            for operator in singular_operators:
                copied_entity = entity
                result = operator.execute_nf([copied_entity])
                entities[d].append(result)
    return entities


def steps_valid(steps):
    test_steps = deepcopy(steps)
    if len(test_steps) == 0:
        return "Empty"
    test_proof = Prover(axioms=all_axioms_to_prove,
                        conditions=test_steps[0]["observation"]["ground_truth"],
                        objectives=test_steps[0]["observation"]["objectives"],
                        prove_direction="backward")
    assert not test_proof.is_proved()
    for step in test_steps:
        for k, op in enumerate(step["input_entities"]):
            # Make sure the root of each operand is in the current graphs
            if is_entity(op):
                assert op.root is not None
                assert op.root.name in [ls.name for ls in test_proof.get_ground_truth() +
                                        test_proof.get_objectives()]
                assert op.root in [ls for ls in step["observation"]["ground_truth"] +
                                   step["observation"]["objectives"]]
            elif is_ls(op):
                assert op.name in [ls.name for ls in test_proof.get_ground_truth() +
                                   test_proof.get_objectives()]
                assert op in [ls for ls in step["observation"]["ground_truth"] +
                              step["observation"]["objectives"]]
            else:
                raise AssertionError("Not allowed type: {}".format(type(op)))
        if step["lemma"].name == "EquivalenceSubstitution":
            op1, op2 = step["input_entities"]

            assembled_operands = list()
            available_ents = []
            for ls in test_proof.get_objectives() + test_proof.get_ground_truth():
                available_ents.extend([ls.ent_dic[key] for key in ls.ent_dic if key != 0])

            replacement1 = test_proof.ls_id2ls[test_proof.ls_name2id[op1.root.name]].ent_dic[op1.index]
            assembled_operands.append(replacement1)
            replacement2 = test_proof.ls_id2ls[test_proof.ls_name2id[op2.root.name]].ent_dic[op2.index]
            assembled_operands.append(replacement2)

        else:
            assembled_operands = list()
            for op in step["input_entities"]:
                for ls in test_proof.get_objectives() + test_proof.get_ground_truth():
                    if is_entity(op) and ls.name == op.root.name:
                        assembled_operands.append(ls.ent_dic[op.index])
                        break
                    elif is_ls(op) and ls.name == op.name:
                        assembled_operands.append(ls)
                        break
            assert len(assembled_operands) == step["lemma"].input_no

        test_proof.apply_theorem(theorem=step["lemma"], operands=assembled_operands, )
    # Make sure the proof is complete when all test_steps are carried out
    assert test_proof.is_proved()


def generate_valid_steps(steps):
    valid_steps = list()
    if steps is None or len(steps) == 0:
        return steps
    test_proof = Prover(axioms=all_axioms_to_prove,
                        conditions=steps[0]["observation"]["ground_truth"],
                        objectives=steps[0]["observation"]["objectives"],
                        prove_direction="backward")
    assert not test_proof.is_proved()
    for step in steps:
        if test_proof.is_proved():
            break
        for op in step["input_entities"]:
            assert (is_entity(op) and op.root is not None) or is_ls(op) or is_empty_ls(op)
            assert (is_entity(op) and op.root.name in
                    [ls.name for ls in test_proof.get_ground_truth() + test_proof.get_objectives()]) or \
                   (is_ls(op) and op.name in
                    [ls.name for ls in test_proof.get_ground_truth() + test_proof.get_objectives()] or
                    is_empty_ls(op))

        assert step["lemma"].name != "EquivalenceSubstitution"
        assembled_operands = list()
        for op in step["input_entities"]:
            for ls in test_proof.get_objectives() + test_proof.get_ground_truth():
                if is_entity(op) and ls.name == op.root.name:
                    assembled_operands.append(ls.ent_dic[op.index])
                    break
                elif (is_empty_ls(op) or is_ls(op)) and ls.name == op.name:
                    assembled_operands.append(ls)
                    break
        assert len(assembled_operands) == step["lemma"].input_no

        lemma = step["lemma"]
        operands = assembled_operands
        step = {
            "observation": test_proof.get_observation(),
            "lemma": lemma,
            "input_entities": operands
        }

        valid_steps.append(step)
        test_proof.apply_theorem(theorem=step["lemma"], operands=assembled_operands)

    # Make sure the proof is complete when all steps are carried out
    assert test_proof.is_proved()
    return valid_steps


class Dataset(data_handler.Dataset):
    def __init__(self, trajectories):
        self.trajectories = trajectories
        self.io_tuples = list()
        for trajectory in trajectories:
            datapoint = (
                trajectory['observation'], trajectory['lemma'],
                trajectory['input_entities'],
            )
            self.io_tuples.append(datapoint)

    def __getitem__(self, index):
        return self.io_tuples[index]

    def __len__(self):
        return len(self.io_tuples)

    def get_multiple(self, indices):
        if len(indices) == 1:
            return self.io_tuples[indices[0]],
        else:
            return itemgetter(*indices)(self.io_tuples)

    def merge(self, another_dataset):
        total_trajectories = self.trajectories + another_dataset.trajectories
        self.__init__(total_trajectories)


def valid_combo(combo_names):
    combo_types = [generation_type[name] for name in combo_names]
    equality = 0
    inequality = 0
    transition = 0
    for combo_type in combo_types:
        if combo_type == "Equality":
            equality += 1
        elif combo_type == "Inequality":
            inequality += 1
        elif combo_type == "Transition":
            transition += 1
        else:
            raise NotImplementedError

    if transition > 1:
        return False
    elif transition == 0 and inequality > 0:
        return False
    else:
        return True


def make_up_condition(requirement_type, a, b, no_iv, new_iv=True):
    if new_iv:
        c = Entity(name=chr(ord('c') + no_iv), is_iv=True)
        no_iv += 1
        a_copied = deepcopy(a)
        b_copied = deepcopy(b)
        c_copied = deepcopy(c)

        a_plus_c = necessary_numerical_functions["add"].execute_nf([a_copied, c_copied])
        new_conclusion = necessary_logic_functions[requirement_type].execute_lf([a_plus_c, b_copied])
    else:
        a_copied, b_copied = deepcopy(a), deepcopy(b)
        new_conclusion = necessary_logic_functions[requirement_type].execute_lf([a_copied, b_copied])

    return {
        "conclusion": new_conclusion,
        "no_iv": no_iv
    }


def find_entity_with_name(ls, target_name):
    for ent_id, ent in ls.ent_dic.items():
        if ent.name == target_name:
            return ent
    raise AssertionError


def proof_agrees_with_specs(backward_steps, length, axiom_order, avoid_objective_names):
    if backward_steps is None or len(backward_steps) != length or \
            set([step["lemma"].name for step in backward_steps]) != set(axiom_order) or \
            backward_steps[0]["observation"]["objectives"][0].name in avoid_objective_names:
        return False
    return True


def initialize_prover(ivs=None, ed=None, ent_per_degree=10, degree=0, **kwargs):
    # Initialize the starting entities and the proof
    if ivs is None and ed is None:
        ivs, ed = generate_starting_ents(degree=degree)
    starting_ents = list()
    for k in sorted(ed.keys()):
        starting_ents.extend(random.sample(ed[k], k=min(ent_per_degree, len(ed[k]))))
    random.shuffle(starting_ents)
    ground_truth = list()
    for ent in starting_ents:
        ground_truth.append(necessary_logic_functions["Equivalent"].execute_lf([ent, ent]))
    prover = Prover(axioms=all_axioms, conditions=ground_truth, objectives=[], prove_direction="forward")
    return ivs, prover


def gather_available_entities(proof, entity_length_limit):
    # Gather available entities from premises and proven facts
    entities = list()
    entity_names = set()
    for gt in proof.get_ground_truth():
        for key in sorted(gt.ent_dic.keys()):
            value = gt.ent_dic[key]
            if isinstance(value, Entity) and value.name not in entity_names \
                    and len(value.name) <= entity_length_limit:
                entities.append(value)
                entity_names.add(value.name)
    return entities
