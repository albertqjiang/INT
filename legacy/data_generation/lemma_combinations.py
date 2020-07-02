import random
import itertools
from copy import deepcopy
from proof_system.all_axioms import operator_axioms, equal_axioms, unequal_axioms, general_axioms, all_axioms

random.seed(0)


def lemma_combo_to_string(lemma_combo):
    lemma_name_to_id = {
        all_axioms[key].name: i
        for i, key in enumerate(all_axioms.keys())
    }
    lemma_ids = [lemma_name_to_id[lemma.name] for lemma in lemma_combo]
    lemma_ids.sort()
    lemma_id_strings = [str(lemma_id) for lemma_id in lemma_ids]
    return "_".join(lemma_id_strings)


def subsets_of_lemmas(lemmas):
    """
    Find all subsets of a lemma set.
    :param lemmas: a list of lemmas
    :return:
    """
    lemma_subsets = list()
    for i in range(len(lemmas) + 1):
        lemma_subsets.extend(list(itertools.combinations(lemmas, r=i)))
    return lemma_subsets


def product_to_list(product):
    """
    Convert a product of sets of subsets of lemmas to a list of sets of lemmas
    :param product:
    :return:
    """
    list_of_sets = list()
    for p in product:
        p_set = list()
        for subset in p:
            p_set.extend(list(subset))
        list_of_sets.append(p_set)
    return list_of_sets


def find_operands(entity_list, lemma):
    """
    Given the lemma and the entity, find operands that can be applied with the given lemma.
    :param lemma:
    :return: operands for the lemma
    """
    if lemma.name in all_axioms.keys():
        operands = lemma.infer_operands(entity_list=entity_list)
        if operands is None:
            return None
        elif operands is False:
            perms = list(itertools.permutations(entity_list, r=lemma.input_no))
            random.shuffle(perms)
            return perms
        elif isinstance(operands, list):
            random.shuffle(operands)
            return operands
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def guided_walk(proof, gt, original_lemma_sets, trials=10):
    """
    Given gt, apply lemma from original_lemma_sets for trials to RHS.
    :param proof:
    :param gt:
    :param original_lemma_sets:
    :param trials:
    :return:
    """
    steps = list()
    lhs, rhs = gt.operands
    random.shuffle(original_lemma_sets)
    lemma_sets = [lemma for lemma in original_lemma_sets]
    while len(steps) < trials and len(lemma_sets) > 0:
        lemma = lemma_sets.pop(0)
        entity_ids = proof._parse_entity_ids_from_entity(rhs)
        operands = find_operands(entity_list=[proof.ent_id2ent[ent_id] for ent_id in entity_ids], lemma=lemma)

        if operands is None:
            continue
        elif isinstance(operands, list):
            for operand_trial in operands:
                step = {
                    "observation": proof.get_observation(),
                    "lemma": lemma,
                    "operands": operand_trial
                }

                before_gt_no = len(proof.ls_id2ls)
                result = proof.apply_theorem(theorem=lemma, operands=operand_trial)
                after_gt_no = len(proof.ls_id2ls)
                if after_gt_no > before_gt_no:
                    steps.append(step)

                    gt = proof.ls_id2ls[result["conclusion_ids"][0]]
                    _, rhs = gt.operands

                    random.shuffle(original_lemma_sets)
                    lemma_sets = [lemma for lemma in original_lemma_sets]
                    break
    return steps, proof


if __name__ == "__main__":
    import os
    import pickle
    from pprint import pprint
    from visualization.latex_parse import entity_to_latex

    operator_subsets = [list(operator_axioms.values())]
    print(len(operator_subsets))
    equal_subsets = subsets_of_lemmas(equal_axioms.values())
    print(len(equal_subsets))
    unequal_subsets = subsets_of_lemmas(unequal_axioms.values())
    print(len(unequal_subsets))
    general_subsets = subsets_of_lemmas(general_axioms.values())
    print(len(general_subsets))
    product = itertools.product(operator_subsets, equal_subsets, unequal_subsets, general_subsets)
    list_of_combinations = product_to_list(product)
    print(len(list_of_combinations))
    assert len(list_of_combinations) == \
           len(operator_subsets) * len(equal_subsets) * len(unequal_subsets) * len(general_subsets)

    directory = "../data/expansion_dataset"

    # for lemma_combination in list_of_combinations:
    #     print(lemma_combo_to_string(lemma_combination))

    for file in os.listdir(directory):
        if file.startswith("proof"):
            proof = pickle.load(open(directory + "/" + file, "rb"))

            for lemma_combination in list_of_combinations:
                custom_proof = deepcopy(proof)
                entity = random.choice(custom_proof.get_entities())
                main_gt = entity.root
                steps, proof = guided_walk(proof=custom_proof, gt=main_gt, original_lemma_sets=lemma_combination)
                pprint([len(step["observation"]["ground_truth"]) for step in steps])
                pprint([(step["lemma"].name, [entity_to_latex(operand) for operand in step["operands"]]) for step in
                        steps])
