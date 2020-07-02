import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

from legacy.logic_math import real_number_axioms
from logic.logic import Entity, Proof
from logic.utils import standard_logic_functions
import random
import pickle
import time
from pprint import pprint
from collections import deque

random.seed(0)


def random_search_theorem(entities, total_trials=10, max_entity_size=20, conditions=[], record_trajectories=False,
                          single_degree_capacity=10):
    trajectories = list()
    degree_counter = dict()
    lemmas = [axiom for axiom in real_number_axioms.values()]
    proof = Proof(entities=entities, axioms=lemmas, assumptions=conditions, objectives=list())

    entity_deque = deque(proof.entities, max_entity_size)
    gt_deque = deque(proof.ground_truth, single_degree_capacity)

    conclusion2step = dict()
    steps = dict()
    step_id = 0
    for trial in range(total_trials):
        step_id += 1
        lemma = random.choices(lemmas, k=1)[0]
        input_entities = random.choices(entity_deque, k=lemma.input_no)
        step = {"observation": proof.get_observation(), "lemma": lemma, "input_entities": input_entities}
        result = lemma.execute_th(input_entities)
        assumptions, conclusions, extra_entities = \
            result["Assumptions"], result["Conclusions"], result["ExtraEntities"]

        # Test redundancy
        ground_truth_strings = [gs.name for gs in gt_deque]
        redundant = True
        for conclusion in conclusions:
            if conclusion.name not in ground_truth_strings:
                redundant = False

        if not redundant:
            if not assumptions:
                if record_trajectories:
                    # Linking conclusions to steps
                    for con in conclusions:
                        conclusion2step[con.name] = step_id
                    for objective in conclusions:
                        raw_observation = dict()
                        raw_observation["ground_truth"] = list()
                        raw_observation["lemmas"] = lemmas
                        raw_observation["entities"] = list(entity_deque)
                        raw_observation["objectives"] = [objective]

                        step["observation"] = raw_observation

                for con in conclusions:
                    if not proof.statements_all_valid([con]):
                        gt_deque.append(con)
                        degree_counter[con.degree] = degree_counter.get(con.degree, 0) + 1
                entity_names = [ent.name for ent in entity_deque]
                for e_entity in extra_entities:
                    if e_entity.name not in entity_names:
                        entity_deque.append(e_entity)

            else:
                # Test validity
                valid = True

                for assumption in assumptions:
                    if assumption.name not in ground_truth_strings:
                        valid = False

                if valid:
                    total_degree = max([valid_assump.degree for valid_assump in assumptions]) + 1
                    if record_trajectories:
                        # Linking conclusions to steps
                        for con in conclusions:
                            conclusion2step[con.name] = step_id
                        for objective in conclusions:
                            raw_observation = dict()
                            raw_observation["ground_truth"] = assumptions
                            raw_observation["lemmas"] = lemmas
                            raw_observation["entities"] = list(entity_deque)
                            raw_observation["objectives"] = [objective]

                            step["observation"] = raw_observation

                            tracebacks = [assump for assump in assumptions]
                            while tracebacks:
                                assumption = tracebacks.pop()
                                try:
                                    proving_step = steps[conclusion2step[assumption.name]]
                                    proving_step[0]["observation"]["objectives"] = [objective]
                                    trajectories.append(proving_step[0])
                                    if proving_step[1]:
                                        tracebacks.extend(proving_step[1])
                                except KeyError:
                                    pass

                    for con in conclusions:
                        # Conclusion not redundant
                        if not proof.statements_all_valid([con]):
                            con.degree = total_degree
                            gt_deque.append(con)
                            degree_counter[total_degree] = degree_counter.get(total_degree, 0) + 1
                    entity_names = [ent.name for ent in entity_deque]
                    for e_entity in extra_entities:
                        if e_entity.name not in entity_names:
                            entity_deque.append(e_entity)
    proof.entities = list(entity_deque)
    proof.ground_truth = list(gt_deque)
    if record_trajectories:
        return proof, trajectories
    return proof


if __name__ == "__main__":
    input_no = 3
    entities = [Entity(name="input{}".format(i)) for i in range(1, 1 + input_no)] + \
               [Entity(name="0", is_constant=True), Entity(name="1", is_constant=True)]
    list_entities = list(entities)

    # Generate training trajectories
    number_of_configurations = 1
    max_conditions = 20
    total_trials = 200

    assumption_dict = dict()
    for index in range(number_of_configurations):
        conditions = random.randint(1, max_conditions)
        assumptions = list()
        for _ in range(conditions):
            slf = random.choice(list(standard_logic_functions.values()))
            assumptions.append(slf.execute_lf(random.choices(list_entities, k=slf.input_no)))
        assumption_dict[index] = assumptions

    all_trajectories = list()
    for key, assumptions in assumption_dict.items():
        proof = random_search_theorem(entities, conditions=assumptions, record_trajectories=False,
                                      total_trials=total_trials)
        # all_trajectories.extend(trajectories)
        truth = dict()
        for gt in proof.ground_truth:
            truth[gt.degree] = truth.get(gt.degree, 0) + 1
            print(gt.degree, gt.name)
        pprint(truth)

        pickle.dump(assumptions, open("../data/standard_theorem_dataset/assumptions_{}.p".format(key), "wb"))
        pickle.dump(proof, open("../data/standard_theorem_dataset/simple_proof_{}.p".format(key), "wb"))
        # pickle.dump(trajectories, open("../data/standard_theorem_dataset/trajectories_{}.p".format(key), "wb"))
        # pprint(trajectories)

    pickle.dump(all_trajectories, open("../data/trajectories.p", "wb"))

    # Generate test trajectories
    number_of_configurations = int(number_of_configurations * 0.1)
    assumption_dict = dict()
    for index in range(number_of_configurations):
        conditions = random.randint(1, max_conditions)
        assumptions = list()
        for _ in range(conditions):
            slf = random.choice(list(standard_logic_functions.values()))
            assumptions.append(slf.execute_lf(random.choices(list_entities, k=slf.input_no)))
        assumption_dict[index] = assumptions

    all_trajectories = list()
    for key, assumptions in assumption_dict.items():

        starting_time = time.time()
        proof, trajectories = random_search_theorem(entities, conditions=assumptions, record_trajectories=True,
                                                    total_trials=total_trials)
        all_trajectories.extend(trajectories)
        truth = dict()
        for gt in proof.ground_truth:
            truth[gt.degree] = truth.get(gt.degree, 0) + 1
        pprint(truth)

    pickle.dump(all_trajectories, open("../data/test_trajectories.p", "wb"))

    # for trajectory in all_trajectories:
    #     print(trajectory["observation"]["objectives"][0].name)
