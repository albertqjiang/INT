import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

from legacy.logic_math import real_number_axioms
from logic.logic import Entity, Proof
from logic.utils import standard_logic_functions
import random
from copy import deepcopy

random.seed(0)


def random_search_theorem(entities, total_trials=10, max_entity_size=5, conditions=[], record_trajectories=False):
    trajectories = list()
    degree_counter = dict()
    lemmas = [axiom for axiom in real_number_axioms.values()]
    proof = Proof(entities=entities, axioms=lemmas, assumptions=conditions, objectives=list())
    previous_steps = list()
    # print([lemma.name for lemma in lemmas])

    conclusion2step = dict()
    steps = dict()
    step_id = 0
    for trial in range(total_trials):
        step_id += 1
        lemma = random.choices(lemmas, k=1)[0]
        input_entities = random.choices(proof.entities, k=lemma.input_no)
        result = lemma.execute_th(input_entities)
        assumptions, conclusions, extra_entities = \
            result["Assumptions"], result["Conclusions"], result["ExtraEntities"]

        # Test redundancy
        ground_truth_strings = [gs.name for gs in proof.ground_truth]
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
                        raw_observation["entities"] = proof.entities
                        raw_observation["objectives"] = [objective]
                        step = {"observation": raw_observation, "lemma": lemma, "input_entities": input_entities}
                        step_and_previous_steps = deepcopy(step)
                        step_and_previous_steps["previous_steps"] = previous_steps
                        steps[step_id] = (step_and_previous_steps, None)
                        trajectories.append(step_and_previous_steps)

                        previous_steps.append(step)

                for con in conclusions:
                    if not proof.statements_all_valid([con]):
                        proof.ground_truth.append(con)
                        degree_counter[con.degree] = degree_counter.get(con.degree, 0) + 1
                if len(proof.entities) < max_entity_size:
                    entity_names = [ent.name for ent in proof.entities]
                    for e_entity in extra_entities:
                        if e_entity.name not in entity_names:
                            proof.entities.append(e_entity)

            else:
                # Test validity
                valid = True
                total_degree = 1
                for assumption in assumptions:
                    if assumption.name not in ground_truth_strings:
                        valid = False
                    else:
                        total_degree += assumption.degree

                if valid:
                    if record_trajectories:
                        # Linking conclusions to steps
                        for con in conclusions:
                            conclusion2step[con.name] = step_id
                        for objective in conclusions:
                            raw_observation = dict()
                            raw_observation["ground_truth"] = assumptions
                            raw_observation["lemmas"] = lemmas
                            raw_observation["entities"] = proof.entities
                            raw_observation["objectives"] = [objective]
                            step = {"observation": raw_observation, "lemma": lemma, "input_entities": input_entities}
                            step_and_previous_steps = deepcopy(step)
                            step_and_previous_steps["previous_steps"] = previous_steps
                            previous_steps.append(step)
                            steps[step_id] = (step_and_previous_steps, assumptions)
                            trajectories.append(step_and_previous_steps)

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
                        if not proof.statements_all_valid([con]):
                            con.degree = total_degree
                            proof.ground_truth.append(con)
                            degree_counter[total_degree] = degree_counter.get(total_degree, 0) + 1
                    if len(proof.entities) < max_entity_size:
                        entity_names = [ent.name for ent in proof.entities]
                        for e_entity in extra_entities:
                            if e_entity.name not in entity_names:
                                proof.entities.add(e_entity)
    if record_trajectories:
        return proof, trajectories
    return proof


if __name__ == "__main__":
    input_no = 3
    entities = [Entity(name="input{}".format(i)) for i in range(1, 1 + input_no)] + \
               [Entity(name="0", is_constant=True), Entity(name="1", is_constant=True)]
    list_entities = list(entities)

    number_of_configurations = 20
    max_conditions = 3
    assumption_dict = dict()
    for index in range(number_of_configurations):
        conditions = random.randint(1, max_conditions)
        assumptions = list()
        for _ in range(conditions):
            slf = random.choice(list(standard_logic_functions.values()))
            assumptions.append(slf.execute_lf(random.choices(list_entities, k=slf.input_no)))
        assumption_dict[index] = assumptions

    all_theorems_in_proofs = list()
    for key, assumptions in assumption_dict.items():
        proof = random_search_theorem(entities, conditions=assumptions, record_trajectories=False, total_trials=1000)
        for gt in proof.ground_truth:
            print(gt.name)

        # pickle.dump(proof, open("../data/standard_theorem_dataset/proof_{}.p".format(key), "wb"))
        # pickle.dump(assumptions, open("../data/standard_theorem_dataset/assumptions_{}.p".format(key), "wb"))
