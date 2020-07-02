from copy import deepcopy
import random

from legacy.data_generation.entity_substitution import substitute
from data_generation import steps_valid
from logic.logic import Entity


def steps_with_replacement(steps, new_entity):
    assert len(steps) > 0
    ent_dict = steps[0]["observation"]["objectives"][0].ent_dic
    all_entities = [ent_dict[key] for key in ent_dict if key != 0]
    atomic_entities = [ent for ent in all_entities if ent.iv]
    if len(atomic_entities) == 0:
        return None
    assert len(atomic_entities) > 0
    target_atomic_entity = random.choice(atomic_entities)

    replaced_steps = list()
    for step in steps:
        original_gt_name_to_ind = dict()
        original_obj_name_to_ind = dict()
        step_copy = deepcopy(step)
        for i, gt in enumerate(step["observation"]["ground_truth"]):
            original_gt_name_to_ind[gt.name] = i
            gt_replaced = substitute(None, gt, target_atomic_entity, new_entity)
            step_copy["observation"]["ground_truth"][i] = gt_replaced
        for i, gt in enumerate(step["observation"]["objectives"]):
            original_obj_name_to_ind[gt.name] = i
            gt_replaced = substitute(None, gt, target_atomic_entity, new_entity)
            step_copy["observation"]["objectives"][i] = gt_replaced

        assembled_operands = []
        for i, ent in enumerate(step["input_entities"]):
            if ent.root.name in original_gt_name_to_ind:
                ent_replaced = step_copy["observation"]["ground_truth"][original_gt_name_to_ind[ent.root.name]].ent_dic[
                    ent.index]
            elif ent.root.name in original_obj_name_to_ind:
                ent_replaced = step_copy["observation"]["objectives"][original_obj_name_to_ind[ent.root.name]].ent_dic[
                    ent.index]
            else:
                raise NotImplementedError
            assembled_operands.append(ent_replaced)
        step_copy["input_entities"] = assembled_operands

        for i, ls in enumerate(step_copy["observation"]["ground_truth"] + step_copy["observation"]["objectives"]):
            (step_copy["observation"]["ground_truth"] + step_copy["observation"]["objectives"])[i].indexing()
            (step_copy["observation"]["ground_truth"] + step_copy["observation"]["objectives"])[i].update_name()
        replaced_steps.append(deepcopy(step_copy))

    return replaced_steps


if __name__ == "__main__":
    new_entity = Entity("h", is_iv=True)
    import pickle
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--f_dir", type=str, required=False,
                    default="../data/random_combination_dataset/")
    ap.add_argument("--b_dir", type=str, required=False,
                    default="../data/renaming_dataset/")
    ap.add_argument("--f_no", type=str, required=False, default="1")

    args = ap.parse_args()
    f_dir = args.f_dir
    b_dir = args.b_dir
    if not os.path.exists(b_dir):
        os.mkdir(b_dir)

    forward_no = args.f_no
    if os.path.isdir(f_dir + "/" + forward_no):
        custom_b_dir = b_dir + "/" + forward_no
        if not os.path.exists(custom_b_dir):
            os.mkdir(custom_b_dir)
        for steps_name in os.listdir(f_dir + "/" + forward_no):
            if steps_name.startswith("steps"):
                steps = pickle.load(open(f_dir + "/" + forward_no + "/" + steps_name, "rb"))
                new_steps = steps_with_replacement(steps, new_entity)
                if new_steps is not None:
                    if steps_valid(new_steps) != "Empty":
                        pickle.dump(new_steps, open(b_dir + "/" + forward_no + "/" + steps_name, "wb"))
