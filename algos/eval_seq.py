import os
import pickle
import json
import argparse

from proof_system.prover import Prover
from data_generation.seq_prefairseq import chared_to_string
from proof_system.all_axioms import all_axioms_to_prove


def process_fairseq_generation_file(generation_file_folder):
    generation_file_path = os.path.join(generation_file_folder, "generate-test.txt")
    indices_to_contents = dict()
    with open(generation_file_path) as fhand:
        for line in fhand.readlines():
            line = line.strip()
            if line.startswith("S-"):
                line_split = line.split()
                line_no = int(line_split[0].lstrip("S-"))
                unchared_string = chared_to_string(line_split[1:])
                indices_to_contents[line_no] = {"source": unchared_string}
            elif line.startswith("H-"):
                line_split = line.split()
                line_no = int(line_split[0].lstrip("H-"))
                unchared_string = chared_to_string(line_split[2:])
                indices_to_contents[line_no]["hypothesis"] = unchared_string

    source_path, hypothesis_path = os.path.join(generation_file_folder, "test.src"), \
        os.path.join(generation_file_folder, "test.hyp")
    with open(source_path, "w") as src_out, open(hypothesis_path, "w") as hyp_out:
        for index in sorted(indices_to_contents.keys()):
            if "source" not in indices_to_contents[index] or "hypothesis" not in indices_to_contents[index]:
                continue
            src_out.write(indices_to_contents[index]["source"])
            src_out.write("\n")
            hyp_out.write(indices_to_contents[index]["hypothesis"])
            hyp_out.write("\n")
    return source_path, hypothesis_path


def execute_according_to_dictionary(prover, dictionary):
    source = prover.parser.observation_to_source(prover.get_observation())
    if source not in dictionary:
        return "no source"
    target = dictionary[source]
    try:
        result = prover.apply_theorem_seq_style(target)
    except Exception:
        return "invalid"
    return result["progress"]


def eval_seq_model(src_path, hyp_path, test_problems_path, eval_fingerprint, dump_path):
    # TODO: save n best responses
    # Load generated question-answer pairs
    sources_to_targets = dict()
    with open(src_path) as src_r, open(hyp_path) as tgt_r:
            for src_line, tgt_line in zip(src_r.readlines(), tgt_r.readlines()):
                sources_to_targets[src_line.strip()] = tgt_line.strip()

    # Load test problems to evaluate
    test_problems = pickle.load(open(os.path.join(test_problems_path, "test_problems.pkl"), "rb"))
    proofs_closed = 0

    for test_problem in test_problems:
        test_step_1 = test_problem[1]
        test_prover = Prover(axioms=all_axioms_to_prove,
                             conditions=test_step_1["observation"]["ground_truth"],
                             objectives=test_step_1["observation"]["objectives"],
                             prove_direction="backward")
        for _ in range(10):
            progress = execute_according_to_dictionary(test_prover, sources_to_targets)
            if not progress:
                break
            if test_prover.is_proved():
                proofs_closed += 1
                break
    json.dump(proofs_closed/len(test_problems), open(os.path.join(dump_path, eval_fingerprint), "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a model output against a set of test problems.')
    parser.add_argument('--fairseq-generate-path', '-fgp')
    parser.add_argument('--dump-path', help='The dump path')
    parser.add_argument('--test-problems-path', '-tpp', help='The test problems path')
    parser.add_argument("--fingerprint", "-f")
    args = parser.parse_args()

    src_path, hyp_path = process_fairseq_generation_file(args.fairseq_generate_path)
    eval_seq_model(src_path, hyp_path, args.test_problems_path, args.fingerprint, args.dump_path)
