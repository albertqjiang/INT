from copy import deepcopy
import pickle
import random

from legacy.connection_prover_exp.connection_prover_backward import ConnectionProverBack as Proof
from proof_system.all_axioms import all_axioms
from legacy.data_generation.backward_conversion import forward_to_backward
from data_generation import steps_valid, find_entity_with_name


def entity_name_in_ls_namelist(entity_name, ls_namelist):
    in_namelist = False
    for ls_name in ls_namelist:
        if (len(entity_name) == 1 and entity_name in ls_name.split()) or (
                len(entity_name) != 1 and entity_name in ls_name):
            in_namelist = True
            break
    return in_namelist


def expansion_to_inequality(steps):
    translated_steps1 = []
    translated_steps2 = []
    p = Proof(axioms=all_axioms,
              conditions=steps[-1]["observation"]["ground_truth"],
              objectives=steps[0]["observation"]["objectives"])
    obj = p.get_objectives()[0]
    p.apply_theorem(theorem=steps[-1]["lemma"],
                    operands=steps[-1]["input_entities"])

    lemma = all_axioms["SquareGEQZero"]
    operands = obj.operands[0].operands
    print(obj.name)
    translated_steps1.append(
        deepcopy({
            "observation": p.get_observation(),
            "lemma": lemma,
            "input_entities": operands
        })
    )
    result = p.apply_theorem(theorem=lemma,
                             operands=operands)
    conclusion = p.ls_id2ls[result["conclusion_ids"][0]]

    lemma = all_axioms["EquivalenceSubstitution"]
    operands = [conclusion.operands[0], obj.operands[1]]
    translated_steps1.append(
        deepcopy({
            "observation": p.get_observation(),
            "lemma": lemma,
            "input_entities": operands
        })
    )
    result = p.apply_theorem(theorem=lemma,
                             operands=operands)
    new_obj = p.ls_id2ls[result["conclusion_ids"][0]]
    old_obj = deepcopy(new_obj)
    # print(old_obj)

    move = True
    while move and new_obj.operands[0].recent_numerical_function and \
            new_obj.operands[0].recent_numerical_function.name == "add":
        if random.random() <= 0.5:
            move = False
        if move:
            lemma = all_axioms["IneqMoveTerm"]
            operands = new_obj.operands[0].operands + [new_obj.operands[1]]
            translated_steps2.append(
                deepcopy(
                    {
                        "observation": p.get_observation(),
                        "lemma": lemma,
                        "input_entities": operands
                    }
                )
            )
            result = p.apply_theorem(theorem=lemma,
                                     operands=operands)
            new_obj = p.ls_id2ls[result["conclusion_ids"][0]]

    for i in range(len(translated_steps1)):
        new_obj_copy = deepcopy(new_obj)
        translated_steps1[i]["observation"]["objectives"] = [new_obj_copy]
    for i in range(len(translated_steps2)):
        new_obj_copy = deepcopy(new_obj)
        translated_steps2[i]["observation"]["objectives"] = [new_obj_copy]
    # for j in range(len(steps)):
    #     old_obj_copy = deepcopy(old_obj)
    #     steps[j]["observation"]["objectives"] = [old_obj_copy]
    # print(len(steps))

    translated_steps = list(reversed(translated_steps2)) + forward_to_backward(steps) + list(
        reversed(translated_steps1))

    # for step in translated_steps:
    #     print("Ground truth")
    #     for gt in step["observation"]["ground_truth"]:
    #         print(logic_statement_to_latex(gt))
    #     print("Objective", logic_statement_to_latex(step["observation"]["objectives"][0]))
    #     print("Lemma", step["lemma"].name)
    #     print("Operands", [entity_to_latex(ent) for ent in step["input_entities"]])
    #     print()

    # # Test the objective can be proven by the translated steps
    # test_proof = Proof(axioms=all_axioms, conditions=[], objectives=[deepcopy(new_obj)])
    # for step in translated_steps:
    #     test_proof.apply_theorem(step["lemma"], step["input_entities"])
    # assert test_proof.is_proved()
    # print("Length of the steps", len(translated_steps))
    real_steps = []
    test_proof = Proof(axioms=all_axioms, conditions=[], objectives=[deepcopy(new_obj)])
    print("Unequal", new_obj.name)
    assert not test_proof.is_proved()
    count = 0
    while len(translated_steps) > 0 and count < 10000 and not test_proof.is_proved():
        count += 1
        step = translated_steps.pop(0)
        # print("Step", step["lemma"].name)
        ls_names = [ls.name for ls in test_proof.get_objectives() + test_proof.get_ground_truth()]
        if step["lemma"].name == "EquivalenceSubstitution":
            can_be_replaced = True
            op1, op2 = step["input_entities"]
            if not op1.root.name in ls_names:
                can_be_replaced = False
            if not entity_name_in_ls_namelist(op2.name, ls_names):
                can_be_replaced = False

            if not can_be_replaced:
                translated_steps.append(step)
            else:
                replaced_operands = list()
                replaced_operands.append(test_proof.ls_id2ls[test_proof.ls_name2id[op1.root.name]].ent_dic[op1.index])
                for ls in test_proof.get_ground_truth() + test_proof.get_objectives():
                    ls_name = ls.name
                    if (len(op2.name) == 1 and op2.name in ls_name.split()) or \
                            (len(op2.name) != 1 and op2.name in ls_name):
                        ent = find_entity_with_name(ls, op2.name)
                        replaced_operands.append(ent)
                        break
                step = {
                    "observation": test_proof.get_observation(),
                    "lemma": step["lemma"],
                    "input_entities": replaced_operands
                }
                real_steps.append(deepcopy(step))
                test_proof.apply_theorem(step["lemma"], replaced_operands)
        else:
            can_be_replaced = True
            for op in step["input_entities"]:
                if not entity_name_in_ls_namelist(op.name, ls_names):
                    can_be_replaced = False
                    break
            if can_be_replaced:
                replaced_operands = list()
                for op in step["input_entities"]:
                    for ls in test_proof.get_objectives() + test_proof.get_ground_truth():
                        ls_name = ls.name
                        if (len(op.name) == 1 and op.name in ls_name.split()) or \
                                (len(op.name) != 1 and op.name in ls_name):
                            ent = find_entity_with_name(ls, op.name)
                            replaced_operands.append(ent)
                            break
                step = {
                    "observation": test_proof.get_observation(),
                    "lemma": step["lemma"],
                    "input_entities": replaced_operands
                }
                real_steps.append(deepcopy(step))
                test_proof.apply_theorem(step["lemma"], replaced_operands)
            else:
                translated_steps.append(step)

            # assert op.root in step["observation"]["ground_truth"] + step["observation"]["objectives"]
    if count >= 10000:
        print(test_proof.ls_id2ls[test_proof.objective_ids[0]].name)
        print("BAD")
        return []
        # raise AssertionError
    else:
        assert test_proof.is_proved()

    # print(new_obj.name)
    return real_steps


if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--e_dir", type=str, required=False,
                    default="/u/ajiang/Projects/ineqSolver/Inequality/data/expansion_dataset")
    ap.add_argument("--i_dir", type=str, required=False,
                    default="/u/ajiang/Projects/ineqSolver/Inequality/data/inequality_dataset")

    args = ap.parse_args()
    e_dir = args.e_dir
    i_dir = args.i_dir
    if not os.path.exists(i_dir):
        os.mkdir(i_dir)

    empty_steps = list()
    for d_name in os.listdir(e_dir):
        if os.path.isdir(e_dir + "/" + d_name):
            if not os.path.exists(i_dir + "/" + d_name):
                os.mkdir(i_dir + "/" + d_name)
            for f_name in os.listdir(e_dir + "/" + d_name):
                if f_name.startswith("steps"):
                    # print(d_name)
                    # print(f_name)
                    steps = pickle.load(open(e_dir + "/" + d_name + "/" + f_name, "rb"))

                    assert len(steps[0]["observation"]["ground_truth"]) == 0
                    translated_steps = expansion_to_inequality(steps)
                    if len(translated_steps) > 0:
                        steps_valid(translated_steps)
                        pickle.dump(translated_steps, open(i_dir + "/" + d_name + "/" + f_name, "wb"))
                    else:
                        # This shouldn't happen
                        empty_steps.append(i_dir + "/" + d_name + "/" + f_name)
                        os.remove(e_dir + "/" + d_name + "/" + f_name)

    print(len(empty_steps))
