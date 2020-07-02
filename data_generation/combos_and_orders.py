import random
import argparse
import json
import os
from copy import deepcopy

from proof_system.all_axioms import generation_type, axiom_sets
from data_generation.utils import valid_combo

random.seed(0)


# Generation orders and proof orders are different because of the existence of AdditionZero and MultiplicationOne.
# We calculate the proof order here because that is what we want to test generalization upon
def calculate_kl(generation_axiom_order):
    k = len(set(generation_axiom_order))
    l = len(generation_axiom_order)
    return k, l


# Divide axioms into the three categories: equal, transitive, and unequal according to their transformation types
def divide_axioms(axiom_names_to_use):
    equal_theorems = [axiom for axiom in axiom_names_to_use if generation_type[axiom] == "Equality"]
    transitive_theorems = [axiom for axiom in axiom_names_to_use if generation_type[axiom] == "Transition"]
    unequal_theorems = [axiom for axiom in axiom_names_to_use if generation_type[axiom] == "Inequality"]
    return equal_theorems, transitive_theorems, unequal_theorems


# Generate an order from a possible combination
def generate_order_from_combination(chosen_axioms, application_times, use_tuple=False):
    eq_axioms, transitive_axioms, une_axioms = divide_axioms(chosen_axioms)
    assert len(transitive_axioms) <= 1
    non_transitive_apps = eq_axioms + une_axioms
    random.shuffle(non_transitive_apps)
    if len(chosen_axioms) < application_times:
        additional_apps = random.choices(non_transitive_apps, k=application_times - len(chosen_axioms))
    else:
        additional_apps = []
    applications = transitive_axioms + non_transitive_apps + additional_apps

    eq_applications, transitive_applications, une_applications = divide_axioms(applications)
    applications = eq_applications + transitive_applications + une_applications
    if use_tuple:
        return tuple(applications)
    return applications


# Determine how to use combinations and orders and return the available indices to choose from
def get_combo_order_info(num_axioms, length, train_test, num_order_or_combo=None, **kwargs):
    # Generation of problems should use either axiom combos or axiom orders(XOR)
    use_combos = "combos" in kwargs
    use_orders = "orders" in kwargs
    assert use_combos ^ use_orders

    # Get the correct combinations and available indices for train or test
    k_combos, kl_orders, available_indices = None, None, None
    if use_combos:
        combos = kwargs["combos"]
        k_combos = combos["k{}".format(num_axioms)]
        num_order_or_combo = int(0.9 * len(k_combos)) if num_order_or_combo is None else num_order_or_combo
        available_indices = range(num_order_or_combo) if train_test == "train" \
            else range(num_order_or_combo, len(k_combos))
    if use_orders:
        orders = kwargs["orders"]
        kl_orders = orders["k{}l{}".format(num_axioms, length)]
        num_order_or_combo = int(0.9 * len(kl_orders)) if num_order_or_combo is None else num_order_or_combo
        available_indices = range(num_order_or_combo) if train_test == "train" \
            else range(num_order_or_combo, len(kl_orders))
    return use_combos, use_orders, k_combos, kl_orders, available_indices


# Randomly sample an axiom order, either directly choose one, or first sample an axiom combination and generate an order
def randomize_one_axiom_order(use_combos, use_orders, k_combos, kl_orders, available_indices, length):
    index = random.choice(available_indices)
    if use_combos:
        axiom_order = generate_order_from_combination(deepcopy(k_combos[index]), int(length))
    elif use_orders:
        axiom_order = deepcopy(kl_orders[index])
    else:
        raise NotImplementedError
    assert valid_combo(axiom_order)
    return axiom_order


def generate_combinations_and_orders(available_axioms, max_l, max_k, trial_per_kl=10000):
    """
    Generate valid axiom combinations and orders
    :param available_axioms: list of available axioms to sample from
    :param max_l: maximum length of the orders
    :param max_k: maximum number of unique axioms in the combinations
    :param trial_per_kl: how many combinations and orders to try for each k-l pair
    :return: the axiom combinations and orders
    """
    combinations = dict()
    orders = dict()
    for k in range(1, max_k + 1):
        for l in range(k, max_l + 1):
            k_key = "k{}".format(k)
            kl_key = "k{}l{}".format(k, l)
            for _ in range(trial_per_kl):
                combination = random.sample(list(available_axioms.keys()), k=k)
                if not valid_combo(combination):
                    continue
                try:
                    order = generate_order_from_combination(combination, l, use_tuple=True)
                except IndexError:
                    continue
                if not valid_combo(order):
                    continue

                k_combo_set = combinations.get(k_key, set())
                k_combo_set.add(tuple(sorted(combination)))
                combinations[k_key] = k_combo_set

                kl_order_set = orders.get(kl_key, set())
                kl_order_set.add(order)
                orders[kl_key] = kl_order_set

    # Make the sets created ordered lists to have reproducibility
    # Shuffle them so they can be used for training and evaluation
    for k_key in sorted(combinations):
        combinations[k_key] = sorted(combinations[k_key])
        random.shuffle(combinations[k_key])
    for kl_key in sorted(orders):
        orders[kl_key] = sorted(orders[kl_key])
        random.shuffle(orders[kl_key])

    return combinations, orders


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ComboOrderGen')
    parser.add_argument('-cp', "--combo_path", required=False, type=str,
                        default="data/benchmark/field")
    parser.add_argument('-mk', "--max_k", required=False, type=int,
                        default=7)
    parser.add_argument('-ml', "--max_l", required=False, type=int,
                        default=7)
    parser.add_argument("--trial", required=False, type=int,
                        default=100000)
    args = parser.parse_args()
    if not os.path.isdir(args.combo_path):
        os.makedirs(args.combo_path)

    axiom_set = args.combo_path.split("/")[-1]
    axioms_to_use = axiom_sets[axiom_set]
    axiom_combinations, axiom_orders, = generate_combinations_and_orders(
        axioms_to_use,
        max_k=args.max_k, max_l=args.max_l, trial_per_kl=args.trial
    )
    json.dump(axiom_combinations, open(os.path.join(args.combo_path, "combinations.json"), "w"))
    json.dump(axiom_orders, open(os.path.join(args.combo_path, "orders.json"), "w"))
