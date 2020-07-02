from proof_system.all_axioms import all_axioms
from proof_system.numerical_functions import necessary_numerical_functions
from proof_system.logic_functions import necessary_logic_functions
from legacy.connection_prover_exp.connection_prover_backward import ConnectionProverBack as Proof
from data_generation import all_entities_to_a_degree, steps_valid, make_up_condition
from logic.logic import Entity

import random
from copy import deepcopy


def create_train_steps():
    steps = []
    a = Entity("a", is_iv=True)
    b = Entity("b", is_iv=True)
    c = Entity("c", is_iv=True)
    one = Entity("1", is_constant=True)
    zero = Entity("0", is_constant=True)
    atomic_entities = [a, b, c, zero, one]

    ent_dict = all_entities_to_a_degree(atoms=atomic_entities, operators=necessary_numerical_functions.values(),
                                        degree=1)
    starting_ents = list()
    for degree in ent_dict:
        starting_ents.extend(ent_dict[degree])

    no_iv = sum([1 for ent in atomic_entities if not ent.is_constant])

    makeup = make_up_condition("Equivalent",
                               random.choice(starting_ents),
                               random.choice(starting_ents),
                               no_iv, new_iv=True)
    condition = makeup["conclusion"]
    # no_iv = makeup["no_iv"]

    entity_to_add = random.choice(starting_ents)
    objective = necessary_logic_functions["Equivalent"].execute_lf([
        necessary_numerical_functions["add"].execute_nf([deepcopy(entity_to_add), deepcopy(condition.operands[0])]),
        necessary_numerical_functions["add"].execute_nf([deepcopy(entity_to_add), deepcopy(condition.operands[1])])
    ])

    p = Proof(all_axioms, [condition], [objective])
    lemma = all_axioms["EquivalenceReflexibility"]
    operands = [objective.operands[0].operands[0]]
    step1 = {
        "observation": p.get_observation(),
        "lemma": lemma,
        "input_entities": operands
    }
    steps.append(deepcopy(step1))

    result = p.apply_theorem(lemma, operands)
    conclusion = p.ls_id2ls[result["conclusion_ids"][0]]

    lemma = all_axioms["PrincipleOfEquality"]
    operands = conclusion.operands + condition.operands
    step2 = {
        "observation": p.get_observation(),
        "lemma": lemma,
        "input_entities": operands
    }
    steps.append(deepcopy(step2))

    return steps


def create_test_steps(train_steps):
    test_steps = list()
    p = Proof(all_axioms, conditions=[], objectives=[deepcopy(train_steps[1]["observation"]["ground_truth"][1])])
    lemma = all_axioms["EquivalenceReflexibility"]
    operands = [p.get_objectives()[0].operands[0]]
    step = {
        "observation": p.get_observation(),
        "lemma": lemma,
        "input_entities": operands
    }
    test_steps.append(deepcopy(step))
    return test_steps


if __name__ == "__main__":
    import pickle
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--b_dir", type=str, required=False,
                    default="/u/ajiang/Projects/ineqSolver/Inequality/data/reflexivity_dataset")

    args = ap.parse_args()
    b_dir = args.b_dir
    if not os.path.exists(b_dir):
        os.mkdir(b_dir)
        os.mkdir(b_dir + "/train")
        os.mkdir(b_dir + "/test")
    for i in range(500):
        train_steps = create_train_steps()
        steps_valid(train_steps)
        pickle.dump(train_steps, open(b_dir + "/train/" + "steps_{}.p".format(i), "wb"))
        test_steps = create_test_steps(train_steps)
        steps_valid(test_steps)
        pickle.dump(test_steps, open(b_dir + "/test/" + "steps_{}.p".format(i), "wb"))
