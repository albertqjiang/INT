import random
from copy import deepcopy
import argparse
import pickle
import os
import json
import shutil

from int_environment.data_generation.utils import steps_valid, make_up_condition, \
    generate_valid_steps, Dataset, proof_agrees_with_specs, initialize_prover, gather_available_entities
from int_environment.data_generation.combos_and_orders import get_combo_order_info, randomize_one_axiom_order
from int_environment.data_generation.forward2backward import forward_to_backward
from int_environment.proof_system.all_axioms import all_axioms

random.seed(0)


def get_operands_when_making_up_conditions(how_to_extend, make_up_conclusions, prover, premise_names, no_atom_ents):
    for config in how_to_extend["makeup_config"]:
        new_atom_ent = config.get("new_iv", True)
        makeup = make_up_condition(
            config["requirement_type"],
            config["a"],
            config["b"],
            no_atom_ents, new_iv=new_atom_ent)
        makeup_conclusion, no_atom_ents = makeup["conclusion"], makeup["no_iv"]
        make_up_conclusions.append(makeup_conclusion)
    prover.condition_ids.extend(prover.add_logic_statements(make_up_conclusions))
    prover.update_conditions()
    for con in make_up_conclusions:
        premise_names.append(con.name)

    make_up_conclusions = [prover.ls_id2ls[prover.ls_name2id[con.name]] for con in make_up_conclusions]
    operands = how_to_extend["operand_retrieval"](make_up_conclusions)
    return operands, no_atom_ents


def produce_operands_info_for_ordinary_lemma(prover, lemma, core_gt, entities, do_transform_this_time,
                                             transformed_solutions, premise_names, no_atom_ents):
    how_to_extend = lemma.extend_core_gt(core_gt, entities, transform_gt=do_transform_this_time)
    if how_to_extend["action"]:
        if "original_coding" in how_to_extend:
            original_coding = how_to_extend["original_coding"]
        else:
            original_coding = lemma.original_coding()

        side = None
        if "transformed_side" in how_to_extend:
            do_transform_this_time = True
            side = how_to_extend["transformed_side"]

            if "circle_back_names" in how_to_extend:
                # Avoid commutativity switching things back and forming an infinite loop
                if (lemma.name, tuple(how_to_extend["circle_back_names"])) in transformed_solutions:
                    how_to_extend = lemma.extend_core_gt(core_gt, entities, transform_gt=False)
                    do_transform_this_time = False
                    original_coding = lemma.original_coding()
                else:
                    transformed_solutions.add((lemma.name, tuple(how_to_extend["circle_back_names"])))
        else:
            do_transform_this_time = False

        if "add_condition" in how_to_extend:
            raise NotImplementedError

        # Make up conditions
        make_up_conclusions = list()
        if how_to_extend["makeup"]:
            operands, no_atom_ents = get_operands_when_making_up_conditions(
                how_to_extend, make_up_conclusions, prover, premise_names, no_atom_ents)
        else:
            operands = how_to_extend["operands"]
        return how_to_extend, operands, original_coding, side, do_transform_this_time, no_atom_ents, make_up_conclusions
    else:
        return None


def produce_operands_info_for_substitution(prover, result, how_to_extend, make_up_conclusions):
    # Apply substitution axiom, if required
    if "substitution_retrieval" in how_to_extend:
        proof_conclusion = prover.ls_id2ls[result["conclusion_ids"][0]]
        # If substitution is required after theorem application
        operands = how_to_extend["substitution_retrieval"](make_up_conclusions, proof_conclusion)
        lemma = all_axioms["EquivalenceSubstitution"]
        return lemma, operands
    else:
        return None, None


def apply_ordinary_lemma(probability_no_transform, transform_gt, prover, lemma, core_gt, entities,
                         transformed_solutions, premise_names, no_atom_ents, steps):
    # Decide whether to do transformation or not according to a 1-e/e probability
    # Default e=0
    do_transform_this_time = False if random.random() < probability_no_transform else transform_gt
    action_info = produce_operands_info_for_ordinary_lemma(prover, lemma, core_gt, entities, do_transform_this_time,
                                                           transformed_solutions, premise_names, no_atom_ents)
    # There's no action that can be performed at this time, therefore an invalid problem
    if action_info is None:
        return
    how_to_extend, operands, original_coding, side, do_transform_this_time, no_atom_ents, make_up_conclusions, \
        = action_info
    # Apply ordinary.sh axiom step
    step = {
        "observation": prover.get_observation(),
        "lemma": lemma,
        "input_entities": operands,
        "original_coding": original_coding,
        "transform_gt": do_transform_this_time,
        "transformed_side": side,
        "custom_function": how_to_extend.get("custom_function", None)
    }
    result, core_gt = proceed_step(step, prover, steps)
    return result, core_gt, how_to_extend, make_up_conclusions


def apply_substitution(prover, steps, result, how_to_extend, make_up_conclusions):
    # Apply substitution step
    if result is None:
        return
    lemma, operands = produce_operands_info_for_substitution(prover, result, how_to_extend, make_up_conclusions)
    if lemma is None:
        return False, False
    # Need to do substitution
    step = {
        "observation": prover.get_observation(),
        "lemma": lemma,
        "input_entities": operands
    }
    result, core_gt = proceed_step(step, prover, steps, mode="substitution")

    # The substitution fails, therefore problem invalid
    if result is None:
        return
    return core_gt, result


def proceed_step(step, prover, steps, mode="ordinary.sh"):
    """
    Given a proof step, apply it to the prover, and add it to all the steps collected, if it proceeds successfully
    :return: the result of the proof step application, and the new core ground truth
    """
    result = prover.apply_theorem(step["lemma"], step["input_entities"])
    interpretation = prover.interpret_result(result)
    if interpretation == "REWARD_ASSUMPTION_INVALID":
        raise AssertionError
    elif interpretation == "REWARD_DUPLICATED_RESULTS":
        if mode == "ordinary.sh":
            return None, None
        core_gt = False
    elif interpretation == "REWARD_THEOREM_PROCEEDED":
        steps.append(step)
        core_gt = prover.ls_id2ls[result["conclusion_ids"][0]]
    else:
        print(interpretation)
        raise NotImplementedError
    return result, core_gt


def add_premises_to_each_step(prover, steps, core_gt):
    # Get conditions
    conditions = [prover.ls_id2ls[con_id] for con_id in prover.condition_ids]
    # Add made up conditions to each step
    if len(steps) > 0:
        # For evaluation purpose
        for s, step in enumerate(steps):
            gt_names = {
                gt.name for gt in step["observation"]["ground_truth"]
            }
            for con in conditions:
                if con.name not in gt_names:
                    steps[s]["observation"]["ground_truth"].append(con)
        for j in range(len(steps)):
            core_gt_copy = core_gt
            steps[j]["observation"]["objectives"].append(core_gt_copy)
    return conditions


def get_a_forward_problem(atom_ents, prover, axiom_order, no_atom_ent_max=20, no_node_max=2000, entity_length_limit=20,
                          transform_gt=True, probability_no_transform=0.0, **kwargs):
    """
    Generate a theorem and its forward proof
    :param atom_ents: atomic entities(like a, b, c)
    :param prover: the forward prover used to write the proof
    :param axiom_order: the order in which axioms should be applied
    :param no_atom_ent_max: the limit of atomic entities used
    :param no_node_max: the limit of graph nodes allowed
    :param entity_length_limit: the limit of the length of entities
    :param transform_gt: whether to allow transforming the core ground truth or only extending it
    :param probability_no_transform: the probability of not doing transform_gt
    :param kwargs: other keyword arguments
    :return: the forward proof steps of the theorem generated
    """
    # Initialize prover and starting conditions
    steps = []
    premise_names = list()
    transformed_solutions = set()
    used_atom_ents, forward_prover = deepcopy(atom_ents), deepcopy(prover)
    core_gt = random.choice(forward_prover.get_ground_truth())
    no_atom_ent = sum([1 for ent in atom_ents if not ent.is_constant])

    for axiom_name in axiom_order:

        # Apply ordinary.sh lemma
        lemma = all_axioms[axiom_name]
        entities = gather_available_entities(forward_prover, entity_length_limit)
        lemma_application = apply_ordinary_lemma(
            probability_no_transform, transform_gt, forward_prover, lemma, core_gt, entities,
            transformed_solutions, premise_names, no_atom_ent, steps)

        # The problem is too large in some aspects, abandon this generation
        if no_atom_ent > no_atom_ent_max or len(forward_prover.ent_id2ent) > no_node_max or lemma_application is None:
            return
        result, core_gt, how_to_extend, make_up_conclusions = lemma_application

        # Apply substitution
        substitution_application = \
            apply_substitution(forward_prover, steps, result, how_to_extend, make_up_conclusions)
        if substitution_application is None:
            return
        if not substitution_application[0]:
            continue
        core_gt, result = substitution_application

    # Add made up premises and delete redundant trivial premises
    add_premises_to_each_step(forward_prover, steps, core_gt)
    initial_condition_names = {forward_prover.ls_id2ls[con_id].name
                               for con_id in forward_prover.initial_condition_ids}
    for k, step in enumerate(steps):
        steps[k]["observation"]["ground_truth"] = [gt for gt in step["observation"]["ground_truth"]
                                                   if gt.name not in initial_condition_names]
    return steps


def generate_problem(num_axioms, length, train_test, **kwargs):
    """
    Generate one single theorem and its proof according to requirements
    Return the proof steps, from which the theorem can be easily extracted
    """
    avoid_objective_names = kwargs.get("avoid_objective_names", [])
    # Get combos or orders ready
    use_combos, use_orders, k_combos, kl_orders, available_indices = \
        get_combo_order_info(num_axioms, length, train_test, **kwargs)
    # Initialize the atomic entities and the proof
    atom_ents, prover = initialize_prover(**kwargs)

    done = False
    returned_steps = None
    while not done:
        axiom_order = randomize_one_axiom_order(use_combos, use_orders, k_combos, kl_orders, available_indices, length)
        forward_steps = get_a_forward_problem(atom_ents, prover, axiom_order, **kwargs)
        if forward_steps is None:
            continue

        try:
            # TODO: Temporary fix, need to investigate
            # Convert the proof to backward and validate it
            returned_steps = generate_valid_steps(forward_to_backward(forward_steps))
        except TypeError:
            continue
        # Check if the proof generated satisfies the specifications given
        if not proof_agrees_with_specs(returned_steps, length, axiom_order, avoid_objective_names):
            continue
        done = True
    steps_valid(returned_steps)
    return returned_steps


def generate_multiple_problems(num_axioms, length, num_probs, **kwargs):
    """
    Generate multiple theorems and proofs and return the tuple (datasets, problems)

    :param num_axioms: the number of unique axioms in the proofs generated

    :param length: the length of the proofs generated

    :param num_probs: how many theorems and proofs to generate

    :param kwargs: keyword arguments to provide additional specifications for the problems generated

        :keyword train_test: str, optional
        Could be "train" or "test", specifies which partition of the axiom combinations or orders to use, default: "train"

        :keyword combos: dict, optional
        The dictionary containing all the available combinations to use in generation, cannot appear at the same time as orders

        :keyword orders: dict, optional
        The dictionary containing all the available orders to use in generation, cannot appear at the same time as combos

        :keyword avoid_objective_names: list, optional
        The list containing all the objective names we wish to avoid during generation. Use this to prevent having test problems, default: []

        :keyword ivs: list, optional
        Individual variables to initialize the prover, default: [a, b, c]

        :keyword ed: dict, optional
        Entity dictionary of the initial prover, default: [0:a, 1:b, 2:c]

        :keyword ent_per_degree: int, optional
        The number of entities to sample from each degree, default: 10

        :keyword degree: int, optional
        The maximum degree of entities sampled, default: 0

        :keyword no_atom_ent_max: int, optional
        Maximum number of atomic entities, default: 20

        :keyword no_node_max: int, optional
        Maximum number of nodes in the graphs, default: 2000

        :keyword entity_length_limit: int, optional
        Maximum length of entities in characters, default: 20

        :keyword transform_gt: bool, optional
        Whether to enable transform_gt, default: True

        :keyword probability_no_transform: float, optional
        Probability of not doing transform_gt in each step, default 0.0

    :return: tuple (Datasets, Problems)

        Datasets is a list containing:
            One dataset with the keyword "all" contains all the proof steps generated, randomly shuffled
            One dataset with the keyword "all_first" contains the first proof steps of each problem, randomly shuffled

        Problems is a list, each element of which is all the proof steps for an individual theorem
    """
    separate_problems = []
    all_steps = []
    all_first_steps = []

    for i in range(num_probs):
        if i % 100 == 0:
            print("Problem {}".format(len(separate_problems) + 1))
        steps = generate_problem(num_axioms, length, **kwargs)
        all_steps.extend(steps)
        all_first_steps.append(steps[0])
        separate_problems.append(steps)

    random.shuffle(all_steps)
    random.shuffle(all_first_steps)
    all_steps_dataset = Dataset(all_steps)
    all_first_steps_dataset = Dataset(all_first_steps)

    multiple_problem_datasets = {
        "all": all_steps_dataset,
        "all_first": all_first_steps_dataset,
    }

    return multiple_problem_datasets, separate_problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mode generator')
    parser.add_argument('--orders_path',
                        default="data/benchmark/field")
    parser.add_argument('--dump_path', '-dp',
                        default="/scratch/hdd001/home/ajiang/int_rerun/problems")
    parser.add_argument('-k', type=int)
    parser.add_argument('-l', type=int)
    parser.add_argument('--degree', type=int, default=0)
    parser.add_argument('--num_probs', type=int, default=100)
    args = parser.parse_args()

    orders = json.load(open(os.path.join(args.orders_path, "orders.json"), "r"))

    kl_dir = os.path.join(args.dump_path, args.orders_path.split("/")[-1], "k={}_l={}".format(args.k, args.l))
    if not os.path.isdir(kl_dir):
        os.makedirs(kl_dir)

    datasets, problems = generate_multiple_problems(num_axioms=args.k, length=args.l,
                                                    num_probs=args.num_probs, train_test="train",
                                                    orders=orders, degree=args.degree)

    extracted_first_steps = [steps[0] for steps in problems]
    if os.path.isdir(os.path.join(kl_dir, "train")):
        shutil.rmtree(os.path.join(kl_dir, "train"))
    os.mkdir(os.path.join(kl_dir, "train"))
    for i, problem in enumerate(problems):
        pickle.dump(problem, open(os.path.join(kl_dir, "train", "steps_{}.p".format(i + 1)), "wb"))

    pickle.dump(datasets["all"], open(os.path.join(kl_dir, "train.pkl"), "wb"))
    pickle.dump(datasets["all_first"], open(os.path.join(kl_dir, "train_first.pkl"), "wb"))
