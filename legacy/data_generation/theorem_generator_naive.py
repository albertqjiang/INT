import os
import sys

sys.path.insert(0, os.path.abspath('../../..'))

from legacy.logic_math import real_number_axioms, famous_theorems
from logic.logic import Entity, Proof
from logic.utils import standard_logic_functions, standard_numerical_functions
import random
import pickle
import json
from pprint import pprint
from itertools import permutations
from copy import deepcopy

random.seed(0)


def random_search_theorem(entities, total_trials=10, conditions=[], max_entity_size=50, single_degree_capacity=100,
                          debug=False, record_trajectories=False):
    lemmas_trimmed = False

    steps = list()
    name2id = dict()
    degree_counter = dict()
    lemmas = [axiom for axiom in real_number_axioms.values()] + \
             [theorem for theorem in famous_theorems.values()]
    proof = Proof(entities=entities, axioms=lemmas, assumptions=conditions, objectives=list())

    ground_truths = [condition for condition in conditions]

    for trial in range(total_trials):
        if not lemmas_trimmed and degree_counter.get(1, 0) == single_degree_capacity:
            lemmas = [lemma for lemma in lemmas if lemma.assumption_size != 0]
            lemmas_trimmed = True

        valid = False
        while not valid:
            lemma = random.choices(lemmas, k=1)[0]
            valid = lemma.check(ground_truths)[0]

        input_entities = random.choices(entities, k=lemma.input_no)
        result = lemma.execute_th(input_entities)
        assumptions, conclusions, extra_entities = \
            result["Assumptions"], result["Conclusions"], result["ExtraEntities"]

        # Test redundancy
        ground_truth_strings = [gs.name for gs in ground_truths]

        if not assumptions:
            total_degree = 1
            for con in conclusions:
                if con.name not in ground_truth_strings \
                        and degree_counter.get(total_degree, 0) < single_degree_capacity:
                    con.degree = total_degree
                    ground_truths.append(con)
                    degree_counter[total_degree] = degree_counter.get(total_degree, 0) + 1

                    # If record trajectories, add this step in the recorder
                    if record_trajectories:
                        raw_observation = {"ground_truth": assumptions, "lemmas": lemmas, "entities": entities,
                                           "objectives": [con]}
                        step = {"observation": raw_observation, "lemma": lemma, "input_entities": input_entities}
                        steps.append(step)
                        name2id[con.name] = len(steps) - 1

        else:
            # Test validity if some assumptions are required by the theorem
            # Also assign the degree of the already proven ground truths to the assumptions
            valid = True
            for assumption in assumptions:
                try:
                    ground_truth_index = ground_truth_strings.index(assumption.name)
                    assumption.degree = ground_truths[ground_truth_index].degree
                    assumption.premise = ground_truths[ground_truth_index].premise
                except ValueError:
                    valid = False
                    break

            if valid:
                total_degree = sum([valid_assump.degree for valid_assump in assumptions]) + 1

                for con in conclusions:
                    # Conclusion not redundant
                    if con.name not in ground_truth_strings \
                            and degree_counter.get(total_degree, 0) < single_degree_capacity:
                        con.degree = total_degree
                        con.premise = assumptions
                        ground_truths.append(con)
                        degree_counter[total_degree] = degree_counter.get(total_degree, 0) + 1

                        # If trajectories is recorded, put this step and all the steps that lead to it in
                        if record_trajectories:
                            all_ground_truth = [assump for assump in assumptions]
                            all_ground_truth_names = {assump.name for assump in all_ground_truth}

                            # Do DFS search to get all premises in the logic chain
                            initial_search_stack = [assump for assump in assumptions]
                            while initial_search_stack:
                                logic_statement = initial_search_stack.pop()
                                if logic_statement.premise:
                                    initial_search_stack.extend(logic_statement.premise)
                                    for pre in logic_statement.premise:
                                        if pre.name not in all_ground_truth_names:
                                            all_ground_truth.append(pre)
                                            all_ground_truth_names.add(pre.name)

                                    step = deepcopy(steps[name2id[logic_statement.name]])
                                    step["observation"]["objectives"] = [con]
                                    steps.append(step)

                            raw_observation = {"ground_truth": all_ground_truth, "lemmas": lemmas, "entities": entities,
                                               "objectives": [con]}
                            step = {"observation": raw_observation, "lemma": lemma, "input_entities": input_entities}
                            steps.append(step)
                            name2id[con.name] = len(steps) - 1

        # Appending to the existing entities
        if len(entities) < max_entity_size:
            entity_names = [ent.name for ent in entities]
            for e_entity in extra_entities:
                if e_entity.name not in entity_names:
                    entities.append(e_entity)

    proof.entities = entities
    proof.ground_truth = ground_truths + conditions
    for step in steps:
        # Count the distance between the objective and the ground truth
        if step["observation"]["objectives"][0].premise:
            distance = 1
            initial_search_stack = [gt for gt in step["observation"]["objectives"][0].premise]
            gt_names = [gt.name for gt in step["observation"]["ground_truth"]]
            while initial_search_stack:
                distance += 1
                logic_statement = initial_search_stack.pop()
                if logic_statement.name in gt_names:
                    pass
                elif logic_statement.premise:
                    initial_search_stack.extend(logic_statement.premise)
        else:
            distance = 1
        step["observation"]["objectives"][0].degree = distance

        step["observation"]["objectives"][0].premise = None
        for gt in step["observation"]["ground_truth"]:
            gt.premise = None
    return proof, steps


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

    # Generate training trajectories
    number_of_configurations = 1
    total_trials = 100000
    all_conditions = \
        [standard_logic_functions["BiggerOrEqual"].execute_lf(per) for per in permutations(entities, r=2)] + \
        [standard_logic_functions["SmallerOrEqual"].execute_lf(per) for per in permutations(entities, r=2)] + \
        [standard_logic_functions["Equivalent"].execute_lf(per) for per in permutations(entities, r=2)]
    max_conditions = min(10, len(all_conditions))

    assumption_dict = dict()
    for index in range(number_of_configurations):
        conditions = random.randint(1, max_conditions)
        assumptions = random.choices(all_conditions, k=conditions)
        assumption_dict[index] = assumptions

    all_trajectories = list()
    for key, assumptions in assumption_dict.items():
        proof, steps = random_search_theorem(entities, conditions=assumptions, total_trials=total_trials,
                                             record_trajectories=True)
        truth = dict()
        for gt in proof.ground_truth:
            truth[gt.degree] = truth.get(gt.degree, 0) + 1
            # print(gt.degree, gt.name)
        pprint(truth)
        json.dump(truth, open("../data/standard_theorem_dataset/truth_degree{}.json".format(key), "w"))

        pickle.dump(assumptions, open("../data/standard_theorem_dataset/assumptions_{}.p".format(key), "wb"))
        pickle.dump(proof, open("../data/standard_theorem_dataset/proof_{}.p".format(key), "wb"))
        pickle.dump(steps, open("../data/standard_theorem_dataset/steps_{}.p".format(key), "wb"))

        # for step in steps:
        #     print('*'*100)
        #     print(step["observation"]["objectives"][0].name, step["observation"]["objectives"][0].degree)
        #     print(step["lemma"].name, [entity.name for entity in step["input_entities"]])
        #     print([assump.name for assump in step["observation"]["objectives"][0].premise])
