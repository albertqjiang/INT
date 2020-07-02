from legacy.data_generation.random_dataset import *
from data_generation import steps_valid, Dataset, valid_combo, non_refl_steps, make_up_condition, \
    find_entity_with_name
from legacy.data_generation.backward_conversion import backward_convert_directory, forward_to_backward
from proof_system.all_axioms import all_axioms
from collections import OrderedDict

random.seed(0)

a = Entity("a", is_iv=True)
b = Entity("b", is_iv=True)
c = Entity("c", is_iv=True)
independent_variables = [a, b, c]
ent_dict = all_entities_to_a_degree(atoms=independent_variables, operators=necessary_numerical_functions.values(),
                                    degree=2)

operator_and_equal_axioms = {**equal_axioms, **operator_axioms}


def non_sub_axioms(steps):
    non_sub = set()
    for step in steps:
        lemma = step["lemma"]
        if lemma.name != "EquivalenceSubstitution":
            non_sub.add(lemma.name)
    return non_sub


def lemma_sample(available_lemmas, lemma_freqs, used_lemmas):
    unused_lemmas = {lemma_name: lemma for lemma_name, lemma in available_lemmas.items()
                     if lemma_name not in used_lemmas}
    if len(unused_lemmas) == 0:
        unused_lemmas = available_lemmas
    else:
        pass

    keys = list(unused_lemmas.keys())
    # Uniform sampling instead of Boltzmann
    selected_key = random.choice(keys)
    return unused_lemmas[selected_key]


def faster_lemma_sample(available_lemmas, lemma_freqs, used_lemmas):
    unused_lemmas = {lemma_name: lemma for lemma_name, lemma in available_lemmas.items()
                     if lemma_name not in used_lemmas}
    if len(unused_lemmas) == 0:
        unused_lemmas = available_lemmas
    else:
        pass

    keys = list(unused_lemmas.keys())
    neg_freqs = [-lemma_freqs[key] for key in keys]
    probs = [freq - min(neg_freqs) + 1 for freq in neg_freqs]
    selected_key = random.choices(keys, weights=probs, k=1)[0]
    return unused_lemmas[selected_key]


def generate_fixed_length_problem(atom_entities, lemmas, proof, iteration_max, no_independent_v_max, no_nodes_max,
                                  lemma_freqs, entity_length_limit=20):
    used_theorems = []
    # Recording steps
    steps = []
    assumption_names = list()

    key = random.choice(list(sorted(proof.ls_id2ls.keys())))
    core_gt = proof.ls_id2ls[key]
    assert proof.ls_name2id[core_gt.name] in proof.ground_truth_ids

    # Number of independent variables
    no_iv = sum([1 for ent in atom_entities if not ent.is_constant])

    iteration = 0
    while len(steps) < iteration_max \
            and no_iv < no_independent_v_max and len(proof.ent_id2ent) < no_nodes_max:
        # Only select random entities from conditions
        entities = list()
        entity_names = set()
        for gt in proof.get_ground_truth():
            for key in sorted(gt.ent_dic.keys()):
                value = gt.ent_dic[key]
                if key != 0 and value.name not in entity_names and len(value.name) <= entity_length_limit:
                    entities.append(value)
                    entity_names.add(value.name)

        nodes, _ = proof.trace(proof.ls_name2id[core_gt.name])
        used_theorems = list(set(used_theorems))

        lemmas_to_choose_from = OrderedDict()
        for lemma_name in lemmas.keys():
            lemma = lemmas[lemma_name]
            if (len(lemma.assumption_types) == 0 or core_gt.logic_function.name in lemma.assumption_types) \
                    and lemma_name != "EquivalenceSubstitution":
                lemmas_to_choose_from[lemma_name] = lemma

        # TODO: we want to be able to choose lemmas in a more random way.
        # TODO: i.e. we don't always want the first three lemmas to be permutations of ABC
        # TODO: we can have ABABC
        if len(lemmas_to_choose_from) == 0:
            lemma = faster_lemma_sample(lemmas, lemma_freqs, used_lemmas=[])
        else:
            lemma = faster_lemma_sample(lemmas_to_choose_from, lemma_freqs, used_lemmas=used_theorems)

        # Special treatment for Square GEQ Zero axiom
        how_to_extend = lemma.extend_core_gt(core_gt, entities)
        if how_to_extend["action"]:
            if "add_condition" in how_to_extend:
                raise NotImplementedError
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
                        # steps.append(deepcopy(step))
                        steps.append(step)
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
                    make_up_conclusions.append(makeup_conclusion)
                proof.condition_ids.extend(proof.add_logic_statements(make_up_conclusions))
                proof.update_conditions()
                for con in make_up_conclusions:
                    assumption_names.append(con.name)

                make_up_conclusions = [proof.ls_id2ls[proof.ls_name2id[con.name]] for con in make_up_conclusions]
                operands = how_to_extend["operand_retrieval"](make_up_conclusions)
            else:
                operands = how_to_extend["operands"]

            obs = proof.get_observation()
            step = {
                "observation": obs,
                "lemma": lemma,
                "input_entities": operands
            }

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
                    assumption_names.append(assump.name)
                # steps.append(deepcopy(step))
                steps.append(step)
                core_gt = proof.ls_id2ls[result["conclusion_ids"][0]]
                iteration += 1
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
                        assumption_names.append(assump.name)
                    # steps.append(deepcopy(step))
                    steps.append(step)
                    core_gt = proof.ls_id2ls[result["conclusion_ids"][0]]
                    assert proof.ls_name2id[core_gt.name] in proof.ground_truth_ids
                    iteration += 1
                else:
                    raise NotImplementedError

            if "special" in how_to_extend:
                raise NotImplementedError
            else:
                pass
        else:
            pass

    conditions = [proof.ls_id2ls[con_id] for con_id in proof.condition_ids]

    # Add made up conditions to each step
    if len(steps) > 0:
        # For evaluation purpose
        for s, step in enumerate(steps):
            gt_names = {
                gt.name for gt in step["observation"]["ground_truth"]
            }
            for con in conditions:
                if con.name not in gt_names:
                    # steps[s]["observation"]["ground_truth"].append(deepcopy(con))
                    steps[s]["observation"]["ground_truth"].append(con)
        for i in range(len(steps)):
            # core_gt_copy = deepcopy(core_gt)
            core_gt_copy = core_gt
            steps[i]["observation"]["objectives"].append(core_gt_copy)

    # Delete redundant trivial gts
    steps_to_front = []
    # empty_proof = Proof(axioms=all_axioms, conditions=[], objectives=[deepcopy(core_gt)])
    empty_proof = Proof(axioms=all_axioms, conditions=[], objectives=[core_gt])
    con_used = {con_id: False for con_id in proof.initial_condition_ids}
    for con_id in con_used:
        if proof.ls_id2ls[con_id].name in assumption_names:
            # Used as an assumption
            con_used[con_id] = True
            # ent = deepcopy(proof.ls_id2ls[con_id].operands[0])
            ent = proof.ls_id2ls[con_id].operands[0]
            # ground_truth = necessary_logic_functions["Equivalent"].execute_lf([ent, deepcopy(ent)])
            ground_truth = necessary_logic_functions["Equivalent"].execute_lf([ent, ent])
            # print(ground_truth.name)
            steps_to_front.append(
                # deepcopy(
                {
                    "observation": empty_proof.get_observation(),
                    "lemma": all_axioms["EquivalenceReflexibility"],
                    "input_entities": [ground_truth.operands[0]]
                }
                # )

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
                # steps_to_front[s]["observation"]["ground_truth"].append(deepcopy(con))
                steps_to_front[s]["observation"]["ground_truth"].append(con)

    steps = steps_to_front + steps

    con_names = [proof.ls_id2ls[gt_id].name for gt_id in proof.initial_condition_ids]
    for i, step in enumerate(steps):
        gt_proven_and_obj = [gt for gt in step["observation"]["objectives"] + step["observation"]["ground_truth"]
                             if gt.name not in con_names]
        for ls in gt_proven_and_obj:
            assert len(ls.operands) > 0

        for j, op in enumerate(step["input_entities"]):
            if op.root.name in con_names:
                # Change the operand to something in the objective or ground truth instead of in the trivial gt
                for k, ls in enumerate(gt_proven_and_obj):
                    assert len(ls.operands) > 0
                    if (len(op.name) == 1 and op.name in ls.name.split()) or (len(op.name) != 1 and op.name in ls.name):
                        # This is super delicate. Be extremely careful when changing this.
                        replacement = find_entity_with_name(ls, op.name)
                        steps[i]["input_entities"] = \
                            [elem for p, elem in enumerate(steps[i]["input_entities"]) if p != j]
                        steps[i]["input_entities"].insert(j, replacement)
                        break

    for i, step in enumerate(steps):
        for con_id, used in con_used.items():
            if not used:
                for j, gt in enumerate(step["observation"]["ground_truth"]):
                    if gt.name == proof.ls_id2ls[con_id].name:
                        del steps[i]["observation"]["ground_truth"][j]
                        # pass

    # Unit test
    if len(steps) > 0:
        for gt in steps[0]["observation"]["ground_truth"]:
            if gt.operands[0].name == gt.operands[1].name and gt.logic_function.name == "Equivalent":
                print(gt.name)
    else:
        print("Empty steps")

    return conditions, proof.get_ground_truth(), steps, used_theorems, len(proof.ent_id2ent)


def get_train_and_test_combos(online_train_test_split, num_combos, num_axioms):
    split = online_train_test_split
    all_equal_axioms = list(sorted(operator_and_equal_axioms.keys()))
    random.shuffle(all_equal_axioms)
    all_numerical_axioms = {key: operator_and_equal_axioms[key] for key in all_equal_axioms}
    all_combinations = itertools.combinations(all_numerical_axioms.keys(), r=num_axioms)
    valid_combinations = []
    for combination in all_combinations:
        if len(valid_combinations) >= num_combos:
            break
        if valid_combo(combination):
            valid_combinations.append(
                {axiom_name: all_numerical_axioms[axiom_name] for axiom_name in combination}
            )
    random.shuffle(valid_combinations)
    train_combinations = valid_combinations[:int(split * len(valid_combinations))]
    test_combinations = valid_combinations[int(split * len(valid_combinations)):]
    return train_combinations, test_combinations


def specify_problem(axiom_list, length, ivs=None, ed=None, backwards=False):
    if ivs is None:
        ivs, ed = deepcopy((independent_variables, ent_dict))
    if valid_combo(axiom_list):
        # Number of maximum iterations
        iter_max = length
        # Number of maximum independent variables
        no_iv_max = 20
        # Number of maximum entity nodes in the proof
        no_nodes_max = 2000

        lemma_freqs = OrderedDict()
        for lemma in sorted(all_axioms.keys()):
            lemma_freqs[lemma] = 0

        starting_ents = list()
        for k in sorted(ed.keys()):
            starting_ents.extend(random.choices(ed[k], k=10))
        random.shuffle(starting_ents)
        # Core entities to start with

        ground_truth = []
        for ent in starting_ents:
            # ground_truth.append(necessary_logic_functions["Equivalent"].execute_lf([ent, deepcopy(ent)]))
            ground_truth.append(necessary_logic_functions["Equivalent"].execute_lf([ent, ent]))

        # Starting proof
        lemmas_without_sub = [lemma for lemma in axiom_list if lemma != "EquivalenceSubstitution"]
        used_theorems_without_sub = []

        steps = []
        while non_refl_steps(steps) not in [iter_max]:
            P = Proof(axioms=all_axioms, conditions=ground_truth, objectives=[])
            conds, gts, steps, used_theorems, nodes = \
                generate_fixed_length_problem(atom_entities=ivs,
                                              lemmas=axiom_list,
                                              proof=P,
                                              iteration_max=iter_max,
                                              no_independent_v_max=no_iv_max,
                                              no_nodes_max=no_nodes_max,
                                              lemma_freqs=lemma_freqs, )
            print(non_refl_steps(steps), iter_max)
            print(steps[-1]["observation"]["objectives"][0].name)
            if non_refl_steps(steps) > iter_max:
                for step in steps:
                    print(step["lemma"].name)

        if backwards:
            return forward_to_backward(steps)
        else:
            steps = generate_valid_steps(steps)
            return steps
    else:
        return None


def load_online_combo_and_length(combo_path, kl_dirs):
    train_combos_and_lengths = []
    test_combos_and_lengths = []
    for kl in kl_dirs:
        k = kl.split("_")[0][-1]
        l = int(kl[-1])
        train_combos = json.load(open(os.path.join(combo_path, "k={}".format(k), "train_combos.json"), "r"))
        test_combos = json.load(open(os.path.join(combo_path, "k={}".format(k), "test_combos.json"), "r"))

        train_combo_dicts = [{axiom_name: all_axioms[axiom_name] for axiom_name in train_combo}
                             for train_combo in train_combos]
        test_combo_dicts = [{axiom_name: all_axioms[axiom_name] for axiom_name in test_combo}
                            for test_combo in test_combos]

        train_combos_and_lengths.append([train_combo_dicts, l])
        test_combos_and_lengths.append([test_combo_dicts, l])
    return train_combos_and_lengths, test_combos_and_lengths


def initialize_online_combo_and_length(train_dirs, otts, num_combos):
    train_problem_setups = [[int(
        l.split("=")[-1]) for l in v.split("_")] for v in train_dirs]
    train_combos_and_lengths = []
    test_combos_and_lengths = []
    for problem_setup in train_problem_setups:
        num_axioms, length = problem_setup
        train_combos, test_combos = get_train_and_test_combos(
            otts, num_combos, num_axioms)
        train_combos_and_lengths.append([train_combos, length])
        test_combos_and_lengths.append([test_combos, length])
    return train_combos_and_lengths, test_combos_and_lengths


def generate_online_data(train_combos, test_combos, problems, otts, length, backwards=False):
    train_steps = []
    train_first_steps = []
    for train_combo in train_combos:
        for _ in range(problems):
            steps = specify_problem(train_combo, length, backwards=backwards)
            train_steps.extend(deepcopy(steps))
            train_first_steps.extend(deepcopy([steps[0]]))
    random.shuffle(train_steps)
    random.shuffle(train_first_steps)
    train_dataset = Dataset(train_steps)
    train_first_dataset = Dataset(train_first_steps)

    test_steps = []
    test_first_steps = []
    for test_combo in test_combos:
        for _ in range(problems):
            steps = specify_problem(test_combo, length, backwards=backwards)
            test_steps.extend(deepcopy(steps))
            test_first_steps.extend(deepcopy([steps[0]]))
    random.shuffle(test_steps)
    random.shuffle(test_first_steps)
    test_dataset = Dataset(test_steps)
    test_first_dataset = Dataset(test_first_steps)
    return {
        "train": train_dataset,
        "train_first": train_first_dataset,
        "test": test_dataset,
        "test_first": test_first_dataset
    }


def read_and_create_dataset(dataset_dir, split_ratio=1.):
    all_steps = []
    all_files = []
    for f_name in os.listdir(dataset_dir):
        all_files.append(str(os.path.join(dataset_dir, f_name)))

    random.shuffle(all_files)
    first_bunch = all_files[:int(split_ratio * len(all_files))]
    second_bunch = all_files[int(split_ratio * len(all_files)):]

    train_steps = []
    train_first_steps = []
    for file_name in first_bunch:
        steps = pickle.load(open(file_name, "rb"))
        train_steps.extend(deepcopy(steps))
        train_first_steps.append(deepcopy(steps[0]))

    val_steps = []
    val_first_steps = []
    for file_name in second_bunch:
        steps = pickle.load(open(file_name, "rb"))
        val_steps.extend(deepcopy(steps))
        val_first_steps.append(deepcopy(steps[0]))

    return Dataset(train_steps), Dataset(train_first_steps), Dataset(val_steps), Dataset(val_first_steps)


if __name__ == "__main__":
    import itertools
    import random

    random.seed(0)
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--axioms", required=True,
                    help="how many axioms")
    ap.add_argument("-l", "--trajectory_length", required=True,
                    help="length of the trajectories")
    ap.add_argument("-sd", "--saving_directory", required=True,
                    help="where to save all the files")
    ap.add_argument("-bsd", "--backward_saving_directory", required=True,
                    help="where to save the backward files")
    ap.add_argument("-sr", "--split_ratio", required=False, default=0.9,
                    help="split ratio of train and val")
    args = vars(ap.parse_args())

    k = int(args["axioms"])
    length = int(args["trajectory_length"])
    sd = args["saving_directory"]
    bsd = args["backward_saving_directory"]
    split_ratio = args["split_ratio"]

    assert length >= 2 * k - 1

    if not os.path.isdir(sd):
        os.mkdir(sd)

    config_dir = os.path.join(sd, "k={}_l={}".format(k, length))
    if not os.path.isdir(config_dir):
        os.mkdir(config_dir)

    combo_dir = os.path.join(sd, "k={}".format(k))
    if not os.path.isdir(combo_dir):
        os.mkdir(combo_dir)

    how_many_problems_per_combination = 100
    # Check if the combos have already been generated
    if os.path.isfile(os.path.join(combo_dir, "train_combos.json")) and os.path.join(combo_dir, "test_combos.json"):
        train_combos = json.load(open(os.path.join(combo_dir, "train_combos.json"), "r"))
        test_combos = json.load(open(os.path.join(combo_dir, "test_combos.json"), "r"))
    else:
        how_many_combinations = 100

        all_equal_axioms = list(operator_and_equal_axioms.keys())
        random.shuffle(all_equal_axioms)
        all_numerical_axioms = {key: operator_and_equal_axioms[key] for key in all_equal_axioms
                                if key != "EquivalenceReflexibility"}
        all_combinations = itertools.combinations(all_numerical_axioms.keys(), r=k)

        valid_combinations = []
        for combination in all_combinations:
            if len(valid_combinations) == how_many_combinations:
                break
            axioms_to_use = {key: operator_and_equal_axioms[key] for key in combination}
            if valid_combo(axioms_to_use):
                valid_combinations.append([axiom for axiom in axioms_to_use.keys()])
        assert len(valid_combinations) <= how_many_combinations

        random.shuffle(valid_combinations)
        train_combos = valid_combinations[:int(split_ratio * len(valid_combinations))]
        test_combos = valid_combinations[int(split_ratio * len(valid_combinations)):]

        json.dump(train_combos, open(os.path.join(combo_dir, "train_combos.json"), "w"))
        json.dump(test_combos, open(os.path.join(combo_dir, "test_combos.json"), "w"))

    if not os.path.isdir(os.path.join(config_dir, "train")):
        os.mkdir(os.path.join(config_dir, "train"))
    if not os.path.isdir(os.path.join(config_dir, "test")):
        os.mkdir(os.path.join(config_dir, "test"))

    combo_counter = 0
    for train_iter, train_combo in enumerate(train_combos):
        print("Combo {}".format(train_iter))
        print(train_combo)
        axioms_to_use = {key: operator_and_equal_axioms[key] for key in train_combo}
        for i in range(how_many_problems_per_combination):
            print("Problem {}".format(i))
            steps = None
            while steps is None or ((len(non_sub_axioms(steps)) != k) and (len(non_sub_axioms(steps)) != k + 1)):
                if steps is not None:
                    print(len(non_sub_axioms(steps)), k)
                    print(non_sub_axioms(steps))
                    print("Trial")
                steps = specify_problem(axiom_list=axioms_to_use, length=length)
            steps_valid(steps)
            pickle.dump(steps, open(os.path.join(config_dir, "train", "steps_{}_{}.p".format(combo_counter, i)), "wb"))
        combo_counter += 1

    for test_combo in test_combos:
        axioms_to_use = {key: operator_and_equal_axioms[key] for key in test_combo}
        for i in range(how_many_problems_per_combination):
            steps = None
            while steps is None or ((len(non_sub_axioms(steps)) != k) and (len(non_sub_axioms(steps)) != k + 1)):
                if steps is not None:
                    print("Trial")
                steps = specify_problem(axiom_list=axioms_to_use, length=length)
            steps_valid(steps)
            pickle.dump(steps, open(os.path.join(config_dir, "test", "steps_{}_{}.p".format(combo_counter, i)), "wb"))
        combo_counter += 1

    # Convert forward proofs to backward proofs
    backward_convert_directory(
        forward_directory=sd, backward_directory=bsd, config_str="k={}_l={}/train".format(k, length))
    backward_convert_directory(
        forward_directory=sd, backward_directory=bsd, config_str="k={}_l={}/test".format(k, length))

    # Create pickle files for datasets
    # Forward
    train_dataset, train_first_dataset, val_dataset, val_first_dataset = read_and_create_dataset(
        dataset_dir=os.path.join(config_dir, "train"), split_ratio=split_ratio)
    test_dataset, test_first_dataset, _, _ = read_and_create_dataset(
        dataset_dir=os.path.join(config_dir, "test"))
    pickle.dump(train_dataset, open(os.path.join(config_dir, "train.pkl"), "wb"))
    pickle.dump(train_first_dataset, open(os.path.join(config_dir, "train_first.pkl"), "wb"))
    pickle.dump(val_dataset, open(os.path.join(config_dir, "val.pkl"), "wb"))
    pickle.dump(val_first_dataset, open(os.path.join(config_dir, "val_first.pkl"), "wb"))
    pickle.dump(test_dataset, open(os.path.join(config_dir, "test.pkl"), "wb"))
    pickle.dump(test_first_dataset, open(os.path.join(config_dir, "test_first.pkl"), "wb"))

    # Backward
    back_dir = os.path.join(bsd, "k={}_l={}".format(k, length))
    train_dataset, train_first_dataset, val_dataset, val_first_dataset = read_and_create_dataset(
        dataset_dir=os.path.join(back_dir, "train"), split_ratio=split_ratio)
    test_dataset, test_first_dataset, _, _ = read_and_create_dataset(
        dataset_dir=os.path.join(back_dir, "test"))
    pickle.dump(train_dataset, open(os.path.join(back_dir, "train.pkl"), "wb"))
    pickle.dump(train_first_dataset, open(os.path.join(back_dir, "train_first.pkl"), "wb"))
    pickle.dump(val_dataset, open(os.path.join(back_dir, "val.pkl"), "wb"))
    pickle.dump(val_first_dataset, open(os.path.join(back_dir, "val_first.pkl"), "wb"))
    pickle.dump(test_dataset, open(os.path.join(back_dir, "test.pkl"), "wb"))
    pickle.dump(test_first_dataset, open(os.path.join(back_dir, "test_first.pkl"), "wb"))
