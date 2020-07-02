import itertools
import random
from copy import deepcopy

from proof_system.all_axioms import all_axioms
# from proof_system.connection_prover_lean import ConnectionProverLean
from legacy.connection_prover_exp.connection_prover_backward import ConnectionProverBack
from proof_system.numerical_functions import necessary_numerical_functions
from legacy.data_generation.backward_conversion import forward_to_backward
from data_generation import all_entities_to_a_degree, steps_valid

from logic.logic import Entity


def simplify_entity(entity, axioms):
    lemma_operands = (None, None)
    if entity.recent_numerical_function:
        if entity.recent_numerical_function.name == "mul":
            lemma = axioms["MultiplicationSimplification"]
            operands = entity.operands
            lemma_operands = (lemma, operands)
        elif entity.recent_numerical_function.name == "add":
            lemma = axioms["AdditionSimplification"]
            operands = entity.operands
            lemma_operands = (lemma, operands)
        elif entity.recent_numerical_function.name == "sqr":
            # TODO: implement square simplification
            pass
        else:
            # TODO: implement other simplification
            pass
    return lemma_operands


def expand_entity(entity, axioms, proof=None):
    # assert proof.logic_statement_connected(proof.ls_name2id[entity.root.name], [])
    lemma_operands = (None, None, 0)
    if entity.recent_numerical_function:
        if entity.recent_numerical_function.name == "sqr":
            lemma = axioms["SquareDefinition"]
            operands = entity.operands
            con_number = 0
            lemma_operands = (lemma, operands, con_number)
        elif entity.recent_numerical_function.name == "mul":
            lhs, rhs = entity.operands

            if rhs.recent_numerical_function and rhs.recent_numerical_function.name == "add":
                lemma = axioms["AdditionMultiplicationRightDistribution"]
                operands = [lhs] + rhs.operands
                con_number = 0
                lemma_operands = (lemma, operands, con_number)

            elif lhs.recent_numerical_function and lhs.recent_numerical_function.name == "add":
                lemma = axioms["AdditionMultiplicationLeftDistribution"]
                operands = [rhs] + lhs.operands
                con_number = 0
                lemma_operands = (lemma, operands, con_number)

        return lemma_operands
    else:
        return lemma_operands


def expand_expression(entity_to_expand, count_limit=100, her=True):
    """
    Expand an expression a.
    :param main_gt: main ground truth statement a = a
    :param count_limit: how many expansion steps
    :param her: whether to use her-style augmentation
    :return: the steps to get a = b, where b is the expanded canonical form of a
    """

    proof = ConnectionProverBack(axioms=all_axioms, conditions=[], objectives=[])
    count = 0
    step_strings_taken = set()
    steps = list()
    her_steps = list()

    lemma, operands, con_number = expand_entity(entity_to_expand, all_axioms)
    step = {
        "observation": proof.get_observation(),
        "lemma": lemma,
        "input_entities": operands
    }
    steps.append(deepcopy(step))
    result = proof.apply_theorem(lemma, operands)
    initial_conclusion = [proof.ls_id2ls[index] for index in result["conclusion_ids"]][con_number]

    lemma = all_axioms["EquivalenceReflexibility"]
    operands = [initial_conclusion.operands[1]]
    step = {
        "observation": proof.get_observation(),
        "lemma": lemma,
        "input_entities": operands
    }
    steps.append(deepcopy(step))
    result = proof.apply_theorem(lemma, operands)
    main_gt = [proof.ls_id2ls[index] for index in result["conclusion_ids"]][con_number]
    # print(main_gt.name)

    while count < count_limit:
        count += 1
        current_entity = main_gt.operands[1]
        entity_ids = proof.parse_entity_nodes_from_entity(current_entity)
        # print(entity_ids)
        entities_to_simplify = [proof.ent_id2ent[index] for index in entity_ids]
        while entities_to_simplify:
            entity_to_simplify = entities_to_simplify.pop()
            lemma, operands = simplify_entity(entity_to_simplify, proof.axioms)
            con_number = 0

            # Can't simplify
            if lemma is None:
                # Do expansion
                lemma, operands, con_number = expand_entity(entity_to_simplify, proof.axioms, proof)
            else:
                step_string = lemma.name + " ".join([operand.name for operand in operands])
                # Already taken this step
                if step_string in step_strings_taken:
                    lemma, operands, con_number = expand_entity(entity_to_simplify, proof.axioms, proof)
                # Do simplification
                else:
                    step_strings_taken.add(step_string)
            if lemma:
                # print(lemma.name, [op.name for op in operands])
                step = {"observation": proof.get_observation(),
                        "lemma": lemma,
                        "input_entities": operands}

                gts_before_lemma = len(proof.get_ground_truth())
                result = proof.apply_theorem(theorem=lemma, operands=operands)
                gts_after_lemma = len(proof.get_ground_truth())
                conclusions = [proof.ls_id2ls[index] for index in result["conclusion_ids"]]

                if len(conclusions) > 0:
                    if gts_after_lemma > gts_before_lemma:
                        step_copied = deepcopy(step)
                        steps.append(step_copied)

                    # print(lemma.name)
                    # print([con.name for con in conclusions])
                    equivalent_entity = conclusions[con_number].operands[1]
                    step = {"observation": proof.get_observation(),
                            "lemma": all_axioms["EquivalenceSubstitution"],
                            "input_entities": [entity_to_simplify, equivalent_entity]}

                    gts_before_lemma = len(proof.get_ground_truth())
                    result = proof.apply_theorem(theorem=all_axioms["EquivalenceSubstitution"],
                                                 operands=[entity_to_simplify, equivalent_entity])
                    gts_after_lemma = len(proof.get_ground_truth())

                    # print(entity_to_simplify.root.name)
                    # print(result)
                    assert len(result["conclusion_ids"]) == 1
                    main_gt = proof.ls_id2ls[result["conclusion_ids"][0]]
                    main_gt.indexing()

                    if her:
                        her_patch = [deepcopy(single_step) for single_step in steps]
                        for i in range(len(her_patch)):
                            her_patch[i]["observation"]["objectives"].append(deepcopy(main_gt))
                        her_steps.extend(her_patch)
                    if gts_after_lemma > gts_before_lemma:
                        step_copied = deepcopy(step)
                        for operand in step_copied["input_entities"]:
                            if not proof.logic_statement_connected(proof.ls_name2id[operand.root.name]):
                                print(1)
                                print(operand.root.name)
                        steps.append(step_copied)
                    break

    lemma = all_axioms["EquivalenceTransitivity"]
    operands = initial_conclusion.operands + [main_gt.operands[1]]
    step = {
        "observation": proof.get_observation(),
        "lemma": lemma,
        "input_entities": operands
    }
    steps.append(deepcopy(step))
    result = proof.apply_theorem(lemma, operands)
    # print(result)
    main_gt = [proof.ls_id2ls[index] for index in result["conclusion_ids"]][0]

    for step in steps:
        step["observation"]["objectives"].append(deepcopy(main_gt))

    steps[0]["input_entities"] = steps[0]["observation"]["objectives"][0].operands[0].operands

    proof.objective_ids = [proof.add_logic_statement(main_gt)]
    proof.objectives = [main_gt]
    print(main_gt.name)
    assert proof.is_proved()
    return proof, steps, her_steps


if __name__ == "__main__":
    import pickle
    import os
    import time
    from legacy.data_generation.categorize_problems import categorize_problems

    degree = 1

    # Base entities
    a = Entity("a")
    b = Entity("b")
    c = Entity("c")
    one = Entity("1", is_constant=True)
    zero = Entity("0", is_constant=True)

    atomic_entities = [a, b, c, one, zero]
    complex_entities = all_entities_to_a_degree(
        atoms=atomic_entities,
        operators=necessary_numerical_functions.values(),
        degree=degree
    )[degree]
    for ent in complex_entities:
        print(ent.name)

    total_configurations = len(complex_entities) ** 2
    print("Total number of problems is {}".format(total_configurations))
    time0 = time.time()

    directory = "../data/expansion_dataset"
    all_combo = list(itertools.product(complex_entities, repeat=2))
    random.shuffle(all_combo)

    for i, (entity1, entity2) in enumerate(all_combo):
        if i % 10 == 0 and i > 0:
            average_time = (time.time() - time0) / 10
            print("Problem {} to {} took {} seconds on average.".format(i - 1, i, average_time))
            print("Estimating the whole process will take {} seconds.".format(average_time * total_configurations))
            time0 = time.time()
        try:
            proof1, sim_steps1, _ = expand_expression(entity1, her=False)
            simplified_entity1 = proof1.ls_id2ls[len(proof1.ls_id2ls) - 1].operands[1]
            lemma = all_axioms["EquivalenceSymmetry"]
            operands = [entity1, simplified_entity1]
            step = {
                "observation": proof1.get_observation(),
                "lemma": lemma,
                "input_entities": operands
            }
            sim_steps1.append(deepcopy(step))
            proof1.apply_theorem(lemma, operands)
        except AttributeError:
            simplified_entity1 = entity1

        try:
            proof2, sim_steps2, _ = expand_expression(entity2, her=False)
            simplified_entity2 = proof2.ls_id2ls[len(proof2.ls_id2ls) - 1].operands[1]
            lemma = all_axioms["EquivalenceSymmetry"]
            operands = [entity2, simplified_entity2]
            step = {
                "observation": proof2.get_observation(),
                "lemma": lemma,
                "input_entities": operands
            }
            sim_steps2.append(deepcopy(step))
            proof2.apply_theorem(lemma, operands)
        except AttributeError:
            simplified_entity2 = entity2

        print(simplified_entity1.name, simplified_entity2.name)
        addition = necessary_numerical_functions["add"].execute_nf([simplified_entity1, simplified_entity2])
        # try:
        #     proof, steps, her_steps = expand_expression(addition, her=False)
        # except AttributeError:
        #     raise NotImplementedError
        # print(proof.ls_id2ls[len(proof.ls_id2ls)-1].name)
        total_sum_sqr = necessary_numerical_functions["sqr"].execute_nf([addition])
        proof, steps, _ = expand_expression(total_sum_sqr, her=False)
        obj = proof.ls_id2ls[len(proof.ls_id2ls) - 1]
        # print(obj.name)

        if simplified_entity1.name != entity1.name:
            steps += sim_steps1
            for step in sim_steps1:
                proof.apply_theorem(step["lemma"], step["input_entities"])
            lemma = all_axioms["EquivalenceSubstitution"]
            operands = [obj.operands[0].operands[0].operands[0], entity1]
            step = {
                "observation": proof.get_observation(),
                "lemma": lemma,
                "input_entities": operands
            }
            steps.append(deepcopy(step))
            result = proof.apply_theorem(lemma, operands)
            obj = proof.ls_id2ls[result["conclusion_ids"][0]]
        if simplified_entity2.name != entity2.name:
            steps += sim_steps2
            for step in sim_steps2:
                proof.apply_theorem(step["lemma"], step["input_entities"])
            lemma = all_axioms["EquivalenceSubstitution"]
            operands = [obj.operands[0].operands[0].operands[1], entity2]
            step = {
                "observation": proof.get_observation(),
                "lemma": lemma,
                "input_entities": operands
            }
            steps.append(deepcopy(step))
            result = proof.apply_theorem(lemma, operands)
            obj = proof.ls_id2ls[result["conclusion_ids"][0]]

        for s in range(len(steps)):
            steps[s]["observation"]["objectives"] = [deepcopy(obj)]

        real_steps = forward_to_backward(steps)

        print(entity1.name, entity2.name)
        print(obj.name)

        if not os.path.exists(directory):
            os.makedirs(directory)
        # pickle.dump(proof, open(algos + "/proof_{}.p".format(i), "wb"))
        steps_valid(real_steps)
        pickle.dump(real_steps, open(directory + "/steps_{}.p".format(i), "wb"))
        # pickle.dump(her_steps, open(algos + "/her_steps_{}.p".format(i), "wb"))

    categorize_problems(directory + "/")
