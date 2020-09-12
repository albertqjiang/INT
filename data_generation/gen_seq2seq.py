import argparse
import json
import os
import random

from data_generation.generate_problems import generate_multiple_problems
from visualization.latex_parse import logic_statement_to_latex


random.seed(213)


compact_theorem_name = {
    "AdditionCommutativity": "add_cmmt",
    "AdditionAssociativity": "add_assc",
    "AdditionZero": "add_zero",
    "AdditionSimplification": "add_simp",
    "MultiplicationCommutativity": "mul_cmmt",
    "MultiplicationAssociativity": "mul_assc",
    "MultiplicationOne": "mul_one",
    "MultiplicationSimplification": "mul_simp",
    "AdditionMultiplicationLeftDistribution": "add_mul_l_dist",
    "AdditionMultiplicationRightDistribution": "add_mul_r_dist",
    "SquareDefinition": "sqr_def",
    "EquivalenceSymmetry": "equ_symm",
    "PrincipleOfEquality": "equ_prin",
    "EquMoveTerm": "equ_mv_tm",
    "IneqMoveTerm": "ineq_mv_tm",
    "SquareGEQZero": "sqr_geq_zero",
    "EquivalenceImpliesDoubleInequality": "equ_dbl_ineq",
    "FirstPrincipleOfInequality": "ineq_prin_one",
    "SecondPrincipleOfInequality": "ineq_prin_two"
}


def convert_proof_to_seq2seq(steps, add_theorem_name=True):
    sources, targets = list(), list()
    for i, step in enumerate(steps):
        if len(step["observation"]["objectives"]) != 1:
            continue

        premises = step["observation"]["ground_truth"]
        premises_string = " & ".join([logic_statement_to_latex(premise) for premise in premises])
        source = premises_string + " to " + logic_statement_to_latex(step["observation"]["objectives"][0])
        sources.append(source)

        target = ""
        if add_theorem_name:
            theorem_name = compact_theorem_name[step["lemma"].name]
            target += theorem_name + " "
        target += "| "

        if i != len(steps) - 1:
            next_step = steps[i+1]
            next_premises = next_step["observation"]["ground_truth"]
            next_premises_string = " & ".join([logic_statement_to_latex(premise) for premise in next_premises])
            target += next_premises_string + " to " + \
                      logic_statement_to_latex(next_step["observation"]["objectives"][0])
        else:
            target += "ø"
        targets.append(target)
    return sources, targets


def generate_multiple_seq2seq(multiple_problems, all_sources_to_targets=None, add_theorem_name=True):
    if not all_sources_to_targets:
        all_sources_to_targets = dict()

    for problem in multiple_problems:
        sources, targets = convert_proof_to_seq2seq(problem, add_theorem_name=add_theorem_name)
        for source, target in zip(sources, targets):
            if source in all_sources_to_targets:
                continue
            all_sources_to_targets[source] = target
    return all_sources_to_targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mode generator')
    parser.add_argument('--orders_path',
                        default="/scratch/hdd001/home/ajiang/data/INT/ordered_field")
    parser.add_argument('--dump_path', '-dp', default="/scratch/hdd001/home/ajiang/data/INT/seq")
    parser.add_argument('-k', type=int)
    parser.add_argument('-l', type=int)
    parser.add_argument('--degree', type=int, default=0)
    parser.add_argument('--num_probs', type=int, default=1)
    parser.add_argument('--add_theorem_name', "-atname", default=True, action='store_false')
    args = parser.parse_args()

    orders = json.load(open(os.path.join(args.orders_path, "orders.json"), "r"))
    if args.num_probs > 10000:
        sources_to_targets = None
        for _ in range(int(args.num_probs/1000)):
            datasets, problems = generate_multiple_problems(num_axioms=args.k, length=args.l,
                                                            num_probs=1000, train_test="train",
                                                            orders=orders, degree=args.degree)
            sources_to_targets = generate_multiple_seq2seq(multiple_problems=problems,
                                                           all_sources_to_targets=sources_to_targets,
                                                           add_theorem_name=args.add_theorem_name)
    else:
        datasets, problems = generate_multiple_problems(num_axioms=args.k, length=args.l,
                                                        num_probs=args.num_probs, train_test="train",
                                                        orders=orders, degree=args.degree)
        sources_to_targets = generate_multiple_seq2seq(multiple_problems=problems,
                                                       add_theorem_name=args.add_theorem_name)
    # for source in sorted(sources_to_targets):
    #     print("Source:", source, "; Target:", sources_to_targets[source])
    randomised_keys = list(sources_to_targets.keys())
    random.shuffle(randomised_keys)

    if not os.path.isdir(args.dump_path):
        os.makedirs(args.dump_path)
    with open(os.path.join(args.dump_path, "all.src"), "w") as src_out:
        with open(os.path.join(args.dump_path, "all.tgt"), "w") as tgt_out:
            for key in randomised_keys:
                src_out.write(key)
                src_out.write("\n")
                tgt_out.write(sources_to_targets[key])
                tgt_out.write("\n")
