import json
import math
from copy import deepcopy
from itertools import combinations

from data_generation import all_entities_to_a_degree, steps_valid, valid_combo, make_up_condition, \
    find_entity_with_name
from legacy.data_generation.utils import generate_valid_steps
from logic.logic import Entity
from proof_system.numerical_functions import necessary_numerical_functions
from proof_system.logic_functions import necessary_logic_functions
from proof_system.all_axioms import all_axioms, unequal_and_equal_and_numerical_axioms, generation_type
from legacy.connection_prover_exp.connection_prover_lean import ConnectionProverLean as Proof

import random
import time

random.seed(0)


def lemma_sample_negative_softmax(lemmas, lemma_freqs, used_theorems, epsilon=0.5):
    equal_axioms_all_used = True
    for lemma in lemmas:
        if generation_type[lemma] == "Equality" and lemma not in used_theorems:
            equal_axioms_all_used = False
    if equal_axioms_all_used:
        legal_lemmas = {lemma: lemmas[lemma] for lemma in lemmas if generation_type[lemma] != "Equality"}
    else:
        legal_lemmas = {lemma: lemmas[lemma] for lemma in lemmas if
                        (generation_type[lemma] == "Equality" and lemma not in used_theorems)}
    uni = random.random()
    if uni <= epsilon:
        return random.choice(list(legal_lemmas.values()))
    else:
        legal_lemmas = {lemma_name: lemma for lemma_name, lemma in legal_lemmas.items()
                        if lemma_name not in used_theorems}
        keys = list(legal_lemmas.keys())
        neg_freqs = [-lemma_freqs[key] for key in keys]
        probs = [math.exp(0.1 * freq) for freq in neg_freqs]
        selected_key = random.choices(keys, weights=probs, k=1)[0]
        return legal_lemmas[selected_key]


def maximize_lemma_entropy_core(atom_entities, lemmas, proof, iteration_max, no_independent_v_max, no_nodes_max,
                                lemma_freqs):
    time0 = time.time()
    used_theorems = []
    # print({lemma for lemma in lemmas})
    # Recording steps
    steps = []
    # Shouldn't be used
    step = None

    assumption_names = set()

    core_gt = random.choice(list(proof.ls_id2ls.values()))
    assert proof.ls_name2id[core_gt.name] in proof.ground_truth_ids

    # Number of independent variables
    no_iv = sum([1 for ent in atom_entities if not ent.is_constant])

    result = None
    i = 0
    while i < 3 * len(lemmas) and no_iv < no_independent_v_max and len(proof.ent_id2ent) < no_nodes_max:
        # Only select random entities from conditions
        entities = list()
        for gt in proof.get_ground_truth():
            entities.extend([gt.ent_dic[key] for key in gt.ent_dic if key != 0])

        nodes, _ = proof.trace(proof.ls_name2id[core_gt.name])
        used_theorems = list(set(used_theorems))
        lemmas_without_sub = [lemma for lemma in lemmas if lemma != "EquivalenceSubstitution"]
        used_theorems_without_sub = [lemma for lemma in used_theorems if lemma != "EquivalenceSubstitution"]
        if len(used_theorems_without_sub) == len(lemmas_without_sub):
            print("Time taken:", time.time() - time0)
            break
        else:
            pass

        lemmas_to_choose_from = {lemma_name: lemma for lemma_name, lemma in lemmas.items()
                                 if (len(
                lemma.assumption_types) == 0 or core_gt.logic_function.name in lemma.assumption_types)
                                 and lemma_name != "EquivalenceSubstitution"}
        if len(lemmas_to_choose_from) == 0:
            break
        lemma = lemma_sample_negative_softmax(lemmas_to_choose_from, lemma_freqs, used_theorems=used_theorems)

        # Special treatment for Square GEQ Zero axiom
        if lemma.name != "SquareGEQZero" or \
                (lemma.name == "SquareGEQZero" and "FirstPrincipleOfInequality" in lemmas):
            pass
        else:
            continue
        how_to_extend = lemma.extend_core_gt(core_gt, entities)
        if how_to_extend["action"]:
            if "add_condition" in how_to_extend:
                added = how_to_extend["add_condition"]
                if added.logic_function.name == "Equivalent" and added.operands[0].name == added.operands[1].name:
                    added_lemma = all_axioms["EquivalenceReflexibility"]
                    added_operands = [added.operands[0]]
                    step = {
                        "observation": proof.get_observation(),
                        "lemma": added_lemma,
                        "input_entities": added_operands
                    }
                    result = proof.apply_theorem(added_lemma, added_operands)
                    interpretation = proof.interpret_result(result)
                    if interpretation == "REWARD_THEOREM_PROCEEDED":
                        steps.append(deepcopy(step))
                else:
                    proof.condition_ids.extend(proof.add_logic_statements([how_to_extend["add_condition"]]))
                    proof.update_conditions()
            make_up_conclusions = []
            if how_to_extend["makeup"]:
                # If making up condition is required
                for config in how_to_extend["makeup_config"]:
                    if "new_iv" in config:
                        new_iv = config["new_iv"]
                    else:
                        new_iv = True
                    makeup = make_up_condition(config["requirement_type"],
                                               config["a"],
                                               config["b"],
                                               no_iv, new_iv=new_iv)
                    makeup_conclusion, no_iv = makeup["conclusion"], makeup["no_iv"]
                    # print(makeup_conclusion.name)
                    make_up_conclusions.append(makeup_conclusion)
                # print(lemma)
                # print("Make up conclusion:", logic_statement_to_latex(makeup_conclusion))
                proof.condition_ids.extend(proof.add_logic_statements(make_up_conclusions))
                proof.update_conditions()
                for con in make_up_conclusions:
                    assumption_names.add(con.name)

                make_up_conclusions = [proof.ls_id2ls[proof.ls_name2id[con.name]] for con in make_up_conclusions]
                operands = how_to_extend["operand_retrieval"](make_up_conclusions)
            else:
                operands = how_to_extend["operands"]
                # Shouldn't be used
                makeup_conclusion = False

            obs = proof.get_observation()
            step = {
                "observation": obs,
                "lemma": lemma,
                "input_entities": operands
            }

            # for op in operands:
            #     if op.root not in step["observation"]["ground_truth"] + step["observation"]["objectives"]:
            #         for ls in step["observation"]["ground_truth"] + step["observation"]["objectives"]:
            #             print(ls.name)
            #         print(op.root, [ls for ls in step["observation"]["ground_truth"] + step["observation"]["objectives"]])
            #         raise NotImplementedError

            result = proof.apply_theorem(lemma, operands)

            interpretation = proof.interpret_result(result)
            if interpretation == "REWARD_ASSUMPTION_INVALID":
                raise AssertionError
            elif interpretation == "REWARD_DUPLICATED_RESULTS":
                pass
            elif interpretation == "REWARD_THEOREM_PROCEEDED":
                used_theorems.append(lemma.name)
                lemma_freqs[lemma.name] = lemma_freqs.get(lemma.name, 0) + 1
                assumptions = [proof.ls_id2ls[assump_id] for assump_id in result["assumption_ids"]]
                for assump in assumptions:
                    assumption_names.add(assump.name)
                steps.append(deepcopy(step))
                core_gt = proof.ls_id2ls[result["conclusion_ids"][0]]
                i += 1
            else:
                raise NotImplementedError

            if "substitution_retrieval" in how_to_extend:
                proof_conclusion = proof.ls_id2ls[result["conclusion_ids"][0]]
                # If substitution is required after theorem application
                operands = how_to_extend["substitution_retrieval"](make_up_conclusions, proof_conclusion)
                lemma = all_axioms["EquivalenceSubstitution"]
                step = {
                    "observation": proof.get_observation(),
                    "lemma": lemma,
                    "input_entities": operands
                }
                result = proof.apply_theorem(lemma, operands)
                interpretation = proof.interpret_result(result)
                if interpretation == "REWARD_ASSUMPTION_INVALID":
                    import pdb;
                    pdb.set_trace()
                    raise AssertionError
                elif interpretation == "REWARD_DUPLICATED_RESULTS":
                    pass
                elif interpretation == "REWARD_THEOREM_PROCEEDED":
                    used_theorems.append(lemma.name)
                    lemma_freqs[lemma.name] = lemma_freqs.get(lemma.name, 0) + 1
                    assumptions = [proof.ls_id2ls[assump_id] for assump_id in result["assumption_ids"]]
                    for assump in assumptions:
                        assumption_names.add(assump.name)
                    steps.append(deepcopy(step))
                    core_gt = proof.ls_id2ls[result["conclusion_ids"][0]]
                    assert proof.ls_name2id[core_gt.name] in proof.ground_truth_ids
                    i += 1
                else:
                    raise NotImplementedError

            if "special" in how_to_extend:
                proof_conclusion = proof.ls_id2ls[result["conclusion_ids"][0]]
                lemma = how_to_extend["special"]["lemma"]
                operands = how_to_extend["special"]["operands"](proof_conclusion)
                step = {
                    "observation": proof.get_observation(),
                    "lemma": lemma,
                    "input_entities": operands
                }
                result = proof.apply_theorem(lemma, operands)
                interpretation = proof.interpret_result(result)
                if interpretation == "REWARD_ASSUMPTION_INVALID":
                    import pdb;
                    pdb.set_trace()
                    raise AssertionError
                elif interpretation == "REWARD_DUPLICATED_RESULTS":
                    pass
                elif interpretation == "REWARD_THEOREM_PROCEEDED":
                    lemma_freqs[lemma.name] = lemma_freqs.get(lemma.name, 0) + 1
                    assumptions = [proof.ls_id2ls[assump_id] for assump_id in result["assumption_ids"]]
                    for assump in assumptions:
                        assumption_names.add(assump.name)
                    steps.append(deepcopy(step))
                    core_gt = proof.ls_id2ls[result["conclusion_ids"][0]]
                    assert proof.ls_name2id[core_gt.name] in proof.ground_truth_ids
                    i += 1
                else:
                    raise NotImplementedError
            else:
                pass
        else:
            pass

    conditions = [proof.ls_id2ls[con_id] for con_id in proof.condition_ids]

    # print([logic_statement_to_latex(con) for con in conditions])
    # print("Conditions:")
    # for con in conditions:
    #     print(logic_statement_to_latex(con))
    # print("*" * 100)

    # print([logic_statement_to_latex(con) for con in conditions])
    # Add made up conditions to each step
    if len(steps) > 0:
        # For evaluation purpose
        for s, step in enumerate(steps):
            gt_names = {
                gt.name for gt in step["observation"]["ground_truth"]
            }
            for con in conditions:
                if con.name not in gt_names:
                    # print(logic_statement_to_latex(con))
                    steps[s]["observation"]["ground_truth"].append(deepcopy(con))

            # print([logic_statement_to_latex(ls) for ls in steps[s]["observation"]["ground_truth"]])

        for i in range(len(steps)):
            core_gt_copy = deepcopy(core_gt)
            # core_gt_copy.indexing()
            # core_gt_copy.update_name()
            # print("obj", core_gt_copy.name)
            steps[i]["observation"]["objectives"].append(core_gt_copy)

    # for step in steps:
    #     print("GTS")
    #     for gt in step["observation"]["ground_truth"]:
    #         print(logic_statement_to_latex(gt))
    #     print("*" * 100)

    # Delete redundant trivial gts
    # initial_trivial_gts = [proof.ls_id2ls[con_id] for con_id in proof.initial_condition_ids]
    steps_to_front = []
    empty_proof = Proof(axioms=all_axioms, conditions=[], objectives=[deepcopy(core_gt)])
    con_used = {con_id: False for con_id in proof.initial_condition_ids}
    for con_id in con_used:
        if proof.ls_id2ls[con_id].name in assumption_names:
            # Used as an assumption
            con_used[con_id] = True
            ent = deepcopy(proof.ls_id2ls[con_id].operands[0])
            ground_truth = necessary_logic_functions["Equivalent"].execute_lf([ent, deepcopy(ent)])
            print(ground_truth.name)
            steps_to_front.append(
                deepcopy(
                    {
                        "observation": empty_proof.get_observation(),
                        "lemma": all_axioms["EquivalenceReflexibility"],
                        "input_entities": [ground_truth.operands[0]]
                    }
                )
            )
            empty_proof.apply_theorem(theorem=all_axioms["EquivalenceReflexibility"],
                                      operands=[ground_truth.operands[0]])
    # print(len(steps_to_front))
    for s, step in enumerate(steps_to_front):
        gt_names = {
            gt.name for gt in step["observation"]["ground_truth"]
        }
        for con in conditions:
            if (not (con.logic_function.name == "Equivalent" and con.operands[0].name == con.operands[1].name)) \
                    and con.name not in gt_names:
                # print(logic_statement_to_latex(con))
                steps_to_front[s]["observation"]["ground_truth"].append(deepcopy(con))

    steps = steps_to_front + steps
    # for step in steps:
    #     print("GT number", len(step["observation"]["ground_truth"]))

    con_names = [proof.ls_id2ls[gt_id].name for gt_id in proof.initial_condition_ids]
    # print(con_names)
    for i, step in enumerate(steps):
        gt_proven_and_obj = [gt for gt in step["observation"]["objectives"] + step["observation"]["ground_truth"]
                             if gt.name not in con_names]
        # print([logic_statement_to_latex(ls) for ls in gt_proven_and_obj])
        for ls in gt_proven_and_obj:
            assert len(ls.operands) > 0
        # entity_ids_from_gt_and_obj = []
        # for ls in gt_proven_and_obj:
        #     entity_ids_from_gt_and_obj.extend(proof.parse_entity_nodes_from_ls(ls))
        # entity_ids_from_gt_and_obj = set(entity_ids_from_gt_and_obj)
        # entity_name2ent = []

        # print([logic_statement_to_latex(gt) for gt in proven_gts_and_obj])
        for j, op in enumerate(step["input_entities"]):
            if op.root.name in con_names:
                # Change the operand to something in the objective or ground truth instead of in the trivial gt
                replaced = False
                for k, ls in enumerate(gt_proven_and_obj):
                    assert len(ls.operands) > 0
                    if (len(op.name) == 1 and op.name in ls.name.split()) or (len(op.name) != 1 and op.name in ls.name):
                        # This is super delicate. Be extremely careful when changing this.
                        replacement = find_entity_with_name(ls, op.name)
                        steps[i]["input_entities"] = \
                            [elem for p, elem in enumerate(steps[i]["input_entities"]) if p != j]
                        steps[i]["input_entities"].insert(j, replacement)
                        # assert steps[i]["input_entities"][j]
                        # print(op.name, ls.name)
                        # print(steps[i]["input_entities"][j].name)
                        break

                # root_id = proof.ls_name2id[op.root.name]
                # print(root_id)
                # con_used[root_id] = True
        # print(gt_used)
    for i, step in enumerate(steps):
        for con_id, used in con_used.items():
            # print(con_id, logic_statement_to_latex(proof.ls_id2ls[con_id]), used)
            if not used:
                # print(proof.ls_id2ls[con_id].name)
                for j, gt in enumerate(step["observation"]["ground_truth"]):
                    if gt.name == proof.ls_id2ls[con_id].name:
                        # print("To del:", logic_statement_to_latex(steps[i]["observation"]["ground_truth"][j]))
                        del steps[i]["observation"]["ground_truth"][j]
                        # pass
    # pprint([logic_statement_to_latex(ls) for ls in conditions])

    # for step in steps:
    #     print("Ground truth:", [logic_statement_to_latex(ls) for ls in step["observation"]["ground_truth"]])
    #     print([[ls.ent_dic[key] for key in ls.ent_dic if key != 0] for ls in step["observation"]["ground_truth"]])
    #     print("Objective:", logic_statement_to_latex(step["observation"]["objectives"][0]))
    #     print(step["observation"]["objectives"][0].ent_dic)
    #     print("Lemma:", step["lemma"].name)
    #     print("Operands:", [entity_to_latex(ent) for ent in step["input_entities"]])
    # print(logic_statement_to_latex(core_gt))

    # Unit test
    if len(steps) > 0:
        # try:
        # test_proof = Proof(axioms=all_axioms,
        #                    conditions=steps[0]["observation"]["ground_truth"],
        #                    objectives=steps[0]["observation"]["objectives"])
        pass

        # except AssertionError:
        #     import pdb; pdb.set_trace()
        # test_proof.objective_ids = [test_proof.add_logic_statement(core_gt)]
        # test_proof.objectives = [core_gt]
        # print(test_proof.is_proved())

        # for step in steps:
        #     for op in step["input_entities"]:
        #         try:
        #             # Make sure the root of each operand is in the current graphs
        #             assert op.root in step["observation"]["objectives"] + step["observation"]["ground_truth"]
        #         except AssertionError:
        #             # print(step["lemma"].name)
        #             # print([entity_to_latex(ent) for ent in step["input_entities"]])
        #             # print([logic_statement_to_latex(ent.root) for ent in step["input_entities"]])
        #             # print([logic_statement_to_latex(ls) for ls in step["observation"]["ground_truth"]])
        #             # print(logic_statement_to_latex(step["observation"]["objectives"][0]))
        #             raise AssertionError
        #     # print("step gt", [logic_statement_to_latex(gt) for gt in step["observation"]["ground_truth"]])
        #     # print(logic_statement_to_latex(step["observation"]["objectives"][0]))
        #     # print(step["lemma"].name)
        #     # print([entity_to_latex(ent) for ent in step["input_entities"]])
        #     # print([logic_statement_to_latex(ent.root) for ent in step["input_entities"]])
        #     # print([logic_statement_to_latex(gt) for gt in test_proof.get_ground_truth()])
        #     # print(test_proof.interpret_result(
        #     test_proof.apply_theorem(theorem=step["lemma"], operands=step["input_entities"])
        #     # ))
        # #     print(test_proof.is_proved())
        #
        # # Make sure the proof is complete when all steps are carried out
        # assert test_proof.is_proved()

    if len(steps) > 0:
        gt_no = sum([1 for gt in steps[0]["observation"]["ground_truth"] if gt.operands[0].name == gt.operands[1].name])
        for gt in steps[0]["observation"]["ground_truth"]:
            if gt.operands[0].name == gt.operands[1].name and gt.logic_function.name == "Equivalent":
                print(gt.name)
        # print("Number of trivial ground truth", gt_no)
    else:
        print("Empty steps")

    return conditions, proof.get_ground_truth(), steps, used_theorems, len(proof.ent_id2ent)


def run(num_combos=300, no_lemmas=7, directory="../data/random_combination_dataset"):
    lemma_freqs = {lemma: 0 for lemma in all_axioms}
    proof_lengths = []
    no_nodes = []

    config2theorems = dict()
    config_empty = dict()
    combo_names = set()
    # for i in range(1, len(usable_axioms)):

    i = no_lemmas
    all_axiom_combinations = list(combinations(list(unequal_and_equal_and_numerical_axioms.values()), r=i))
    random.shuffle(all_axiom_combinations)
    config2theorems[i] = dict()
    config_empty[i] = dict()
    print("{} lemmas combo.".format(i))
    custom_directory = directory + "/{}".format(i)
    if not os.path.exists(custom_directory):
        os.makedirs(custom_directory)

    config_saving_index = 0
    for config_index, theorems in enumerate(all_axiom_combinations):
        theorem_names = [theorem.name for theorem in theorems]
        if valid_combo(theorem_names):
            config_saving_index += 1
        else:
            continue
        if config_saving_index > num_combos:
            break

        # Base entity atoms
        a = Entity("a", is_iv=True)
        b = Entity("b", is_iv=True)
        c = Entity("c", is_iv=True)
        one = Entity("1", is_constant=True)
        zero = Entity("0", is_constant=True)
        independent_variables = [a, b, c]
        # atomic_entities = [a, b, c, zero, one]

        # All theorems we can use
        # print(i)
        # print(theorems)
        # for theorem in theorems:
        #     if theorem.name in ["InequalityTransitivity",
        #                         "FirstPrincipleOfInequality", "SecondPrincipleOfInequality"]:
        #         theorems = list(theorems) + [all_axioms["EquivalenceImpliesDoubleInequality"]]
        #         break
        theorems = list(theorems) + [all_axioms["EquivalenceSubstitution"]]
        # theorems = [all_axioms["FirstPrincipleOfInequality"], all_axioms["SquareGEQZero"],
        #             all_axioms["EquivalenceImpliesDoubleInequality"]]
        theorems = list(set(theorems))
        theorems = {
            theorem.name: theorem for theorem in theorems
        }
        theorem_names = tuple(sorted(list((theorems.keys()))))
        combo_names.add(theorem_names)
        config2theorems[i][config_index] = theorem_names
        config_empty[i][config_index] = list()
        # print(theorem_names)

        print_string = []

        for theorem in theorems:
            print_string.append(theorem)

        # Number of maximum iterations
        iter_max = 25

        # Number of maximum independent variables
        no_iv_max = 8
        # Number of maximum entity nodes in the proof
        no_nodes_max = 2000

        problems_per_combo = 100
        saving_index = 0
        # Starting entities are all degree 0 and 1 entities
        ent_dict = all_entities_to_a_degree(atoms=independent_variables,
                                            operators=necessary_numerical_functions.values(),
                                            degree=0)
        starting_ents = list()
        for k in sorted(ent_dict.keys()):
            starting_ents.extend(ent_dict[k])
        random.shuffle(starting_ents)

        ground_truth = []

        # Core entities to start with
        for ent in list(set(starting_ents[:10] + independent_variables)):
            ground_truth.append(necessary_logic_functions["Equivalent"].execute_lf([ent, deepcopy(ent)]))

        # Starting proof
        P = Proof(axioms=all_axioms, conditions=ground_truth, objectives=[])

        steps = []
        while saving_index < problems_per_combo:
            print("Configuration {}".format(config_index))
            print("Problem {}".format(saving_index))
            atomic_entities_copy, P_copy = deepcopy((independent_variables, P))
            conds, gts, steps, used_theorems, nodes = \
                maximize_lemma_entropy_core(atom_entities=atomic_entities_copy,
                                            lemmas=theorems,
                                            proof=P_copy,
                                            iteration_max=iter_max,
                                            no_independent_v_max=no_iv_max,
                                            no_nodes_max=no_nodes_max,
                                            lemma_freqs=lemma_freqs, )
            lemmas_without_sub = [lemma for lemma in theorems if lemma != "EquivalenceSubstitution"]
            used_theorems_without_sub = [lemma for lemma in used_theorems if lemma != "EquivalenceSubstitution"]
            if len(lemmas_without_sub) == len(used_theorems_without_sub) and nodes <= no_nodes_max:
                steps = generate_valid_steps(steps)
                steps_valid(steps)
                pickle.dump(steps, open(custom_directory + "/steps_{}_{}.p".format(config_index, saving_index), "wb"))
                saving_index += 1
                proof_lengths.append(len(steps))
                no_nodes.append(nodes)
            else:
                pass
        print()

    json.dump(config2theorems, open(directory + "/config2theorems_{}.json".format(no_lemmas), "w"))
    # json.dump(config_empty, open(algos + "/config_empty.json", "w"))
    json.dump(lemma_freqs, open(directory + "/lemma_freqs_{}.json".format(no_lemmas), "w"))
    json.dump(proof_lengths, open(directory + "/proof_lengths_{}.json".format(no_lemmas), "w"))
    json.dump(no_nodes, open(directory + "/no_nodes_{}.json".format(no_lemmas), "w"))


def steps_online_generation(lemmas):
    if valid_combo(lemmas):
        # Number of maximum iterations
        iter_max = 25
        # Number of maximum independent variables
        no_iv_max = 8
        # Number of maximum entity nodes in the proof
        no_nodes_max = 2000

        lemma_freqs = {lemma: 0 for lemma in all_axioms}
        a = Entity("a", is_iv=True)
        b = Entity("b", is_iv=True)
        c = Entity("c", is_iv=True)
        independent_variables = [a, b, c]
        ent_dict = all_entities_to_a_degree(atoms=independent_variables,
                                            operators=necessary_numerical_functions.values(),
                                            degree=0)
        starting_ents = list()
        for k in sorted(ent_dict.keys()):
            starting_ents.extend(ent_dict[k])
        random.shuffle(starting_ents)
        # Core entities to start with

        ground_truth = []
        for ent in list(set(starting_ents[:10] + independent_variables)):
            ground_truth.append(necessary_logic_functions["Equivalent"].execute_lf([ent, deepcopy(ent)]))

        # Starting proof
        lemmas_without_sub = [lemma for lemma in lemmas if lemma != "EquivalenceSubstitution"]
        used_theorems_without_sub = []
        P = Proof(axioms=all_axioms, conditions=ground_truth, objectives=[])
        while len(used_theorems_without_sub) != len(lemmas_without_sub):
            atomic_entities_copy, P_copy = deepcopy((independent_variables, P))
            conds, gts, steps, used_theorems, nodes = \
                maximize_lemma_entropy_core(atom_entities=atomic_entities_copy,
                                            lemmas=lemmas,
                                            proof=P_copy,
                                            iteration_max=iter_max,
                                            no_independent_v_max=no_iv_max,
                                            no_nodes_max=no_nodes_max,
                                            lemma_freqs=lemma_freqs, )

            used_theorems_without_sub = [lemma for lemma in used_theorems if lemma != "EquivalenceSubstitution"]
        steps = generate_valid_steps(steps)
        steps_valid(steps)
        return steps
    else:
        return None


if __name__ == "__main__":
    import os
    import pickle

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--num_c", type=int, required=False, default=None)
    ap.add_argument("--dir", type=str, required=False, default=None)
    ap.add_argument("--no_lemmas", type=int, required=False, default=None)

    args = vars(ap.parse_args())
    run(args['num_c'], args['no_lemmas'], args['dir'])
    # run()
