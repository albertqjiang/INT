from copy import deepcopy
import random
import pickle

from proof_system.logic_functions import necessary_logic_functions
from proof_system.numerical_functions import necessary_numerical_functions
from legacy.connection_prover_exp.connection_prover_backward import ConnectionProverBack as Proof
from proof_system.all_axioms import all_axioms
from data_generation import steps_valid
from legacy.data_generation.utils import generate_valid_steps


def change_operator(steps):
    obj = steps[0]["observation"]["objectives"][0]
    if obj.logic_function.name != "Equivalent":
        return None
    else:
        steps_train = deepcopy(steps)
        steps_test = deepcopy(steps)
        obj_train = deepcopy(obj)
        obj_test = deepcopy(obj)

        # Choose the entity to add/multiply to both sides
        irrelevant_entity_choices = list()
        for ls in steps[0]["observation"]["ground_truth"] + steps[0]["observation"]["objectives"]:
            irrelevant_entity_choices.extend([ls.ent_dic[key] for key in ls.ent_dic if key != 0])
        irrelevant_entity = random.choice(irrelevant_entity_choices)

        # Construct training steps
        lhs_train = necessary_numerical_functions["add"].execute_nf([deepcopy(irrelevant_entity),
                                                                     deepcopy(obj_train.operands[0])])
        rhs_train = necessary_numerical_functions["add"].execute_nf([deepcopy(irrelevant_entity),
                                                                     deepcopy(obj_train.operands[1])])
        # c + a = c + b
        new_obj_train = necessary_logic_functions["Equivalent"].execute_lf([lhs_train, rhs_train])
        p_train = Proof(all_axioms, steps_train[-1]["observation"]["ground_truth"], [deepcopy(new_obj_train)])
        p_train.apply_theorem(steps_train[-1]["lemma"], steps_train[-1]["input_entities"])

        lemma = all_axioms["EquivalenceReflexibility"]
        operands = [new_obj_train.operands[0]]
        step = {
            "observation": p_train.get_observation(),
            "lemma": lemma,
            "input_entities": operands
        }
        steps_train.append(deepcopy(step))
        result = p_train.apply_theorem(lemma, operands)
        conclusion = p_train.ls_id2ls[result["conclusion_ids"][0]]

        lemma = all_axioms["EquivalenceSubstitution"]
        operands = [conclusion.operands[1].operands[1], new_obj_train.operands[1].operands[1]]
        step = {
            "observation": p_train.get_observation(),
            "lemma": lemma,
            "input_entities": operands
        }
        steps_train.append(deepcopy(step))
        p_train.apply_theorem(lemma, operands)

        assert p_train.is_proved()

        # Construct test steps
        lhs_test = necessary_numerical_functions["mul"].execute_nf([deepcopy(irrelevant_entity),
                                                                    deepcopy(obj_test.operands[0])])
        rhs_test = necessary_numerical_functions["mul"].execute_nf([deepcopy(irrelevant_entity),
                                                                    deepcopy(obj_test.operands[1])])
        # c * a = c * b
        new_obj_test = necessary_logic_functions["Equivalent"].execute_lf([lhs_test, rhs_test])
        p_test = Proof(all_axioms, steps_train[-1]["observation"]["ground_truth"], [deepcopy(new_obj_test)])
        p_test.apply_theorem(steps_test[-1]["lemma"], steps_test[-1]["input_entities"])

        lemma = all_axioms["EquivalenceReflexibility"]
        operands = [new_obj_test.operands[0]]
        step = {
            "observation": p_test.get_observation(),
            "lemma": lemma,
            "input_entities": operands
        }
        steps_test.append(deepcopy(step))
        result = p_test.apply_theorem(lemma, operands)
        conclusion = p_test.ls_id2ls[result["conclusion_ids"][0]]

        lemma = all_axioms["EquivalenceSubstitution"]
        operands = [conclusion.operands[1].operands[1], new_obj_test.operands[1].operands[1]]
        step = {
            "observation": p_test.get_observation(),
            "lemma": lemma,
            "input_entities": operands
        }
        steps_test.append(deepcopy(step))
        p_test.apply_theorem(lemma, operands)

        assert p_test.is_proved()

        for s in range(len(steps_train)):
            steps_train[s]["observation"]["objectives"] = [deepcopy(new_obj_train)]
        for s in range(len(steps_test)):
            steps_test[s]["observation"]["objectives"] = [deepcopy(new_obj_test)]

        return {
            "train": generate_valid_steps(steps_train),
            "test": generate_valid_steps(steps_test)
        }


if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--f_dir", type=str, required=False,
                    default="/u/ajiang/Projects/ineqSolver/Inequality/data/random_combination_dataset")
    ap.add_argument("--b_dir", type=str, required=False,
                    default="/u/ajiang/Projects/ineqSolver/Inequality/data/operator_changed_dataset")
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
        if not os.path.exists(custom_b_dir + "/train/"):
            os.mkdir(custom_b_dir + "/train/")
        if not os.path.exists(custom_b_dir + "/test/"):
            os.mkdir(custom_b_dir + "/test/")
        for steps_name in os.listdir(f_dir + "/" + forward_no):
            if steps_name.startswith("steps"):
                steps = pickle.load(open(f_dir + "/" + forward_no + "/" + steps_name, "rb"))
                operator_changed_traintest = change_operator(steps)
                if operator_changed_traintest is not None:
                    train_steps = operator_changed_traintest["train"]
                    test_steps = operator_changed_traintest["test"]
                    if steps_valid(train_steps) != "Empty":
                        pickle.dump(train_steps, open(b_dir + "/" + forward_no + "/" + "train/" + steps_name, "wb"))
                    if steps_valid(test_steps) != "Empty":
                        pickle.dump(test_steps, open(b_dir + "/" + forward_no + "/" + "test/" + steps_name, "wb"))
