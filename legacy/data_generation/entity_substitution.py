from copy import deepcopy


def substitute(prover, ls, target_entity, new_entity):
    all_candidate_entities = [ls.ent_dic[key] for key in ls.ent_dic if key != 0]
    for entity in all_candidate_entities:
        if entity.name == target_entity.name:
            parent_node = ls.ent_dic[entity.parent_index]
            for ind, operand in enumerate(parent_node.operands):
                if operand.index == entity.index:
                    new_entity_copy = deepcopy(new_entity)
                    parent_node.operands[ind] = new_entity_copy

                    ls.ent_dic[entity.index] = new_entity_copy
                    new_entity_copy.root = ls
                    new_entity_copy.index = entity.index
            # ls.indexing()
            # ls.update_name()
            # print(ls.name)
    return ls


if __name__ == "__main__":
    """
    NB this augmentation doesn't supply steps
    """
    import pickle
    import os
    import random

    random.seed(0)
    from visualization.latex_parse import logic_statement_to_latex, entity_to_latex

    directory = "../data/expansion_dataset/"
    for file in os.listdir(directory):
        if file.startswith("proof_"):
            base_proof = pickle.load(open(directory + file, "rb"))
            equivalent_proof_name = file.replace("proof", "equivalent_proof")

            new_entity = base_proof.ent_id2ent[0]
            target_entity = base_proof.ent_id2ent[len(base_proof.ent_id2ent) - 1]
            print(entity_to_latex(target_entity), entity_to_latex(new_entity))
            base_proof.objectives[0] = substitute(base_proof, base_proof.objectives[0], target_entity, new_entity)

            pickle.dump(base_proof, open(directory + equivalent_proof_name, "wb"))
