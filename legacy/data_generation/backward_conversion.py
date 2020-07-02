from copy import deepcopy

from proof_system.all_axioms import all_axioms
from legacy.connection_prover_exp.connection_prover_backward import ConnectionProverBack as Proof
from proof_system.logic_functions import necessary_logic_functions
from data_generation import steps_valid, find_entity_with_name
import pickle
import os

from visualization.latex_parse import step_to_latex


def backward_convert_directory(forward_directory, backward_directory, config_str):
    if not os.path.exists(backward_directory):
        os.mkdir(backward_directory)

    if os.path.isdir(os.path.join(forward_directory, config_str)):
        custom_b_dir = backward_directory
        for sub_dir in config_str.split("/"):
            custom_b_dir = os.path.join(custom_b_dir, sub_dir)
            if not os.path.exists(custom_b_dir):
                os.mkdir(custom_b_dir)

        for steps_name in os.listdir(forward_directory + "/" + config_str):
            if steps_name.startswith("steps"):
                steps = pickle.load(open(forward_directory + "/" + config_str + "/" + steps_name, "rb"))
                translated_steps = forward_to_backward(steps)
                steps_valid(translated_steps)
                pickle.dump(translated_steps, open(backward_directory + "/" + config_str + "/" + steps_name, "wb"))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--f_dir", type=str, required=False, default="../data/random_combination_dataset")
    ap.add_argument("--b_dir", type=str, required=False, default="../data/backward_combination_dataset")
    ap.add_argument("--f_no", type=str, required=False, default="1")

    args = ap.parse_args()
    f_dir = args.f_dir
    b_dir = args.b_dir
    forward_no = args.f_no
    backward_convert_directory(f_dir, b_dir, forward_no)


def forward_to_backward(steps, unittest=True):
    if len(steps) == 0:
        return steps
    translated_steps = list()
    # print("Original steps")
    # for step in steps:
    #     print(step["lemma"].name)
    # print("*" * 50)
    # print(steps)
    proof = Proof(
        axioms=all_axioms,
        conditions=steps[0]["observation"]["ground_truth"],
        objectives=steps[0]["observation"]["objectives"]
    )

    # for step in steps:
    #     print(step["lemma"].name)
    #     print([entity_to_latex(ent) for ent in step["input_entities"]])
    #     print([logic_statement_to_latex(ls) for ls in proof.get_ground_truth()])
    #     proof.apply_theorem(step["lemma"], step["input_entities"])

    assert not proof.is_proved()
    # random.shuffle(steps)
    iteration = 0
    while len(steps) > 0 and (not proof.is_proved()):
        iteration += 1
        if iteration >= 100:
            return None
        # print(proof.is_proved())
        step = steps.pop()
        if step["lemma"].name == "EquivalenceSubstitution":
            could_be_replaced = True
            # First op is logic statement sensitive, the second isn't
            gt_and_obj_names = [ls.name for ls in proof.get_ground_truth() + proof.get_objectives()]
            if step["input_entities"][0].root.name not in gt_and_obj_names:
                # print("First element can't be replaced")
                # print(step["input_entities"][0].root.name, gt_and_obj_names)
                could_be_replaced = False
            if step["input_entities"][1].name not in [ent.name for ent in proof.ent_id2ent.values()]:
                # print("Second element can't be replaced")
                # print(step["input_entities"][1].name, [ent.name for ent in proof.ent_id2ent.values()])
                could_be_replaced = False

            if could_be_replaced:
                assembled_operands = list()
                op1, op2 = step["input_entities"]
                replacement1 = proof.ls_id2ls[proof.ls_name2id[op1.root.name]].ent_dic[op1.index]
                assembled_operands.append(replacement1)

                available_ents = []
                for ls in proof.get_objectives() + proof.get_ground_truth():
                    available_ents.extend([ls.ent_dic[key] for key in ls.ent_dic if key != 0])

                replaced = False
                for ent in available_ents:
                    if ent.name == op2.name:
                        replacement2 = ent
                        assembled_operands.append(replacement2)
                        replaced = True
                        break
                assert replaced

                for op in assembled_operands:
                    assert op.root in proof.get_ground_truth() + proof.get_objectives()
                translated_steps.append(

                    {
                        "observation": proof.get_observation(),
                        "lemma": step["lemma"],
                        "input_entities": assembled_operands
                    }

                )
                # print("Old objective:", [proof.ls_id2ls[obj_id].name for obj_id in proof.objective_ids])
                # print(step["lemma"].name)
                result = proof.apply_theorem(theorem=step["lemma"],
                                             operands=assembled_operands)
                # print(result)
                # print("New objective:", [proof.ls_id2ls[obj_id].name for obj_id in proof.objective_ids])
            else:
                pos_to_insert = len(steps) - 1
                if pos_to_insert not in list(range(0, len(steps))):
                    print(pos_to_insert, len(steps))
                while steps[pos_to_insert]["lemma"].name == "EquivalenceSubstitution" and pos_to_insert > 0:
                    pos_to_insert -= 1
                steps.insert(pos_to_insert, step)
        else:
            assembled_operands = list()
            all_replaced = []

            for op in step["input_entities"]:
                replaced = False
                available_ls_names = [ls.name for ls in proof.get_objectives() + proof.get_ground_truth()]
                for ls_name in available_ls_names:
                    ls_id = proof.ls_name2id[ls_name]
                    if (len(op.name) == 1 and op.name in ls_name.split()) or (len(op.name) != 1 and op.name in ls_name):
                        ls = proof.ls_id2ls[ls_id]
                        replacement = find_entity_with_name(ls, op.name)
                        assembled_operands.append(replacement)
                        replaced = True
                        break
                all_replaced.append(replaced)

            if not all(all_replaced):
                steps.insert(0, step)
                continue

            if len(assembled_operands) == step["lemma"].input_no:
                for op in assembled_operands:
                    assert op.root in proof.get_ground_truth() + proof.get_objectives()

                translated_steps.append(
                    {
                        "observation": proof.get_observation(),
                        "lemma": step["lemma"],
                        "input_entities": assembled_operands
                    }
                )
                # print("Old objective:", [proof.ls_id2ls[obj_id].name for obj_id in proof.objective_ids])
                # print(step["lemma"].name)
                # print(step["lemma"].name)
                result = proof.apply_theorem(theorem=step["lemma"],
                                             operands=assembled_operands)
                # print(result)
                assert result is not None
                # print("New objective:", [proof.ls_id2ls[obj_id].name for obj_id in proof.objective_ids])
    # assert proof.is_proved()
    #     for op in step["input_entities"]:
    #         print(step["lemma"])
    #         print(logic_statement_to_latex(op.root))
    #         for ls in proof.get_ground_truth() + proof.get_objectives():
    #             print(logic_statement_to_latex(ls))
    #         assert op.root in proof.get_ground_truth() + proof.get_objectives()
    #     print(proof.apply_theorem(
    #         theorem=step["lemma"],
    #         operands=step["input_entities"]
    #     ))
    #     print(proof.is_proved())
    if unittest:
        assert proof.is_proved()
    return translated_steps


def forward_to_backward_smart_choose(steps, unittest=True):
    if len(steps) == 0:
        return steps
    proof = Proof(
        axioms=all_axioms,
        conditions=steps[0]["observation"]["ground_truth"],
        objectives=steps[0]["observation"]["objectives"]
    )

    translated_steps = list()

    iteration = 0
    while len(steps) > 0 and (not proof.is_proved()):
        iteration += 1
        if iteration >= 100:
            # for step in original_steps:
            #     print(step_to_latex(step))
            # raise NotImplementedError
            print("Failure")
            return None
        step = steps.pop()
        if step["lemma"].name == "EquivalenceSubstitution":
            could_be_replaced = True
            # First op is logic statement sensitive, the second isn't
            gt_and_obj_names = [ls.name for ls in proof.get_ground_truth() + proof.get_objectives()]

            input1, input2, = step["input_entities"]
            a_equal_b = necessary_logic_functions["Equivalent"].execute_lf([input1, input2])
            gt_input1 = deepcopy(input1.root)
            gt_input1.indexing()
            input1_parent = gt_input1.ent_dic[input1.parent_index]
            for k, sibling in enumerate(input1_parent.operands):
                if sibling.index == input1.index:
                    replaced_index = k
                    input1_parent.operands[replaced_index] = deepcopy(input2)
                else:
                    pass
            gt_input1.update_name()
            if gt_input1.name not in gt_and_obj_names:
                could_be_replaced = False
            if input1.name not in [ent.name for ent in proof.ent_id2ent.values()]:
                could_be_replaced = False
            if input1.root.name not in gt_and_obj_names and a_equal_b.name not in gt_and_obj_names:
                could_be_replaced = False

            if could_be_replaced:
                assembled_operands = list()

                replaced = False
                available_ents = []
                for ls in proof.get_objectives() + proof.get_ground_truth():
                    available_ents.extend([ls.ent_dic[key] for key in ls.ent_dic if key != 0])

                for ent in available_ents:
                    if ent.name == input1.name:
                        replacement1 = ent
                        assembled_operands.append(replacement1)
                        replaced = True
                        break
                replacement2 = proof.ls_id2ls[proof.ls_name2id[gt_input1.name]].ent_dic[input1.index]
                assembled_operands.append(replacement2)

                assert replaced

                for op in assembled_operands:
                    assert op.root in proof.get_ground_truth() + proof.get_objectives()
                translated_steps.append(

                    {
                        "observation": proof.get_observation(),
                        "lemma": step["lemma"],
                        "input_entities": assembled_operands
                    }

                )
                result = proof.apply_theorem(theorem=step["lemma"],
                                             operands=assembled_operands)
            else:
                pos_to_insert = len(steps) - 1
                if pos_to_insert not in list(range(0, len(steps))):
                    print(pos_to_insert, len(steps))
                while steps[pos_to_insert]["lemma"].name == "EquivalenceSubstitution" and pos_to_insert > 0:
                    pos_to_insert -= 1
                steps.insert(pos_to_insert, step)
        else:
            assembled_operands = list()

            obj_and_gts = proof.get_objectives() + proof.get_ground_truth()

            structured_replacement = False
            combined_entity_name = step["lemma"].combined_entity_name(step["input_entities"])
            if combined_entity_name is not None:
                if isinstance(combined_entity_name, list):
                    for i, single_name in enumerate(combined_entity_name):
                        if structured_replacement:
                            break
                        for ls in obj_and_gts:
                            if structured_replacement:
                                break
                            if single_name in ls.name:
                                structured_replacement = True
                                combined_entity = find_entity_with_name(ls, single_name)
                                assembled_operands = step["lemma"].combined_entity_to_operands(i, combined_entity)

                elif isinstance(combined_entity_name, str):
                    for ls in obj_and_gts:
                        if combined_entity_name in ls.name:
                            structured_replacement = True
                            combined_entity = find_entity_with_name(ls, combined_entity_name)
                            assembled_operands = step["lemma"].combined_entity_to_operands(None, combined_entity)
                            break
                else:
                    raise NotImplementedError

            if not structured_replacement and combined_entity_name is not None:
                # print([entity_to_latex(single_name, True) for single_name in combined_entity_name])
                # print(step["lemma"].name)
                # print(logic_statement_to_latex(step["observation"]["objectives"][0]))
                # print([entity_to_latex(op) for op in step["input_entities"]])
                # print([logic_statement_to_latex(ls) for ls in obj_and_gts])
                # raise NotImplementedError
                pos_to_insert = len(steps) - 1
                if pos_to_insert not in list(range(0, len(steps))):
                    print(pos_to_insert, len(steps))
                while pos_to_insert > 0 and steps[pos_to_insert]["lemma"].name != "EquivalenceSubstitution":
                    pos_to_insert -= 1
                steps.insert(pos_to_insert, step)

            if combined_entity_name is None:
                all_replaced = []
                for op in step["input_entities"]:
                    replaced = False
                    for ls in obj_and_gts:
                        ls_name = ls.name
                        if (len(op.name) == 1 and op.name in ls_name.split()) or (
                                len(op.name) != 1 and op.name in ls_name):
                            replacement = find_entity_with_name(ls, op.name)
                            assembled_operands.append(replacement)
                            replaced = True
                            break
                    all_replaced.append(replaced)

                if not all(all_replaced):
                    steps.insert(0, step)
                    continue

            if len(assembled_operands) == step["lemma"].input_no:
                for op in assembled_operands:
                    assert op.root in proof.get_ground_truth() + proof.get_objectives()

                translated_steps.append(
                    {
                        "observation": proof.get_observation(),
                        "lemma": step["lemma"],
                        "input_entities": assembled_operands
                    }
                )
                result = proof.apply_theorem(theorem=step["lemma"],
                                             operands=assembled_operands)
                assert result is not None
    if unittest:
        assert proof.is_proved()

    for step in translated_steps:
        print(step_to_latex(step))
    return translated_steps
