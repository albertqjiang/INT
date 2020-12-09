import argparse
import pickle
import os
import numpy as np

from proof_system.graph_seq_conversion import Parser

proof_parser = Parser()


def collect_problems(problems_path):
    problems = list()
    used_objectives = set()
    for file_name in os.listdir(problems_path):
        if file_name.startswith("problems"):
            batch_problems = pickle.load(open(os.path.join(problems_path, file_name), "rb"))
            for problem in batch_problems:
                if problem[-1]["observation"]["objectives"][0].name not in used_objectives:
                    problems.append(problem)
                    used_objectives.add(problem[-1]["observation"]["objectives"][0].name)
    print("{} problems in total.".format(len(problems)))
    return problems


def problems_to_dreamer_form(problems):
    all_episodes = list()
    for problem in problems:
        episode = list()
        next_step = problem[0]
        next_state_string, next_action_string = proof_parser.parse_proof_step_to_seq(next_step)
        for i in range(len(problem)):
            state_string, action_string = next_state_string, next_action_string
            state = proof_parser.seq_string_chared(state_string)
            print(action_string)

            if action_string.split()[-1].isdigit():
                action = proof_parser.seq_string_chared(" ".join(action_string.split()[:-1])) \
                         + " <space> {}".format(action_string.split()[-1])
            else:
                action = proof_parser.seq_string_chared(action_string)
            print(action)
            reward = 1 if i == len(problem) - 1 else 0

            if reward == 1:
                validity = 1
                next_state_string = "emp"
            else:
                next_step = problem[i+1]
                next_state_string, next_action_string = proof_parser.parse_proof_step_to_seq(next_step)
                if next_state_string == state_string:
                    validity = 0
                else:
                    validity = 1
            next_state = proof_parser.seq_string_chared(next_state_string)

            episode.append((state, action, next_state, reward, validity))
        all_episodes.append(
            np.array(episode,
                     dtype=[('state', object), ('action', object), ('next_state', object),
                            ('reward', '<f4'), ('validity', '<i4')]))
    return np.array(all_episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mode generator')
    parser.add_argument('--dump_path', '-dp')
    args = parser.parse_args()

    all_problems = collect_problems(args.dump_path)
    episodes = problems_to_dreamer_form(all_problems)
    np.savez_compressed(os.path.join(args.dump_path, "episodes"), a=episodes)
