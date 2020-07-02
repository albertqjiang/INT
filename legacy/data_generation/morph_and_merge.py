import random
from copy import deepcopy

from proof_system.all_axioms import all_axioms
from legacy.connection_prover_exp.connection_prover_backward import ConnectionProverBack as Proof
from legacy.data_generation.expansion_generator import simplify_entity


def morph_objective(steps):
    steps_to_append = list()
    p = Proof(all_axioms, steps[-1]["observation"]["ground_truth"], steps[-1]["observation"]["objectives"])
    obj = p.get_objectives()[0]

    while random.random() > 0.1:
        lhs = obj.operands[0]
        ents = [lhs.ent_dic[key] for key in lhs.ent_dic if key != 0]
        ent = random.choice(ents)
        lemma, operands = simplify_entity(ent, all_axioms)
        if lemma is not None:
            step = {
                "observation": p.get_observation(),
                "lemma": lemma,
                "input_entities": operands
            }
            steps_to_append.append(deepcopy(step))
            result = p.apply_theorem(lemma, operands)
            conclusion = p.ls_id2ls[result["conclusion_ids"][0]]

            lemma = all_axioms["EquivalenceSubstitution"]
            operands = [ent, conclusion.operands[1]]
            step = {
                "observation": p.get_observation(),
                "lemma": lemma,
                "input_entities": operands
            }
            steps_to_append.append(deepcopy(step))
            result = p.apply_theorem(lemma, operands)
            obj = p.ls_id2ls[result["conclusion_ids"][0]]

        else:
            pass
