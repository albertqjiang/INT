import os
import sys
import pickle
import json
import torch

sys.path.insert(0, os.path.abspath('../../../..'))

from legacy.helper_functions.utils import load_config
from legacy.rl_algos.goal_based.naive_her import AllInOneNet, objectives_to_coupled_proofs_and_provers


def evaluate(configuration, theorems_per_degree=10):
    model = torch.load(configuration["saving_dir"] + "model.pt")
    evaluation_performance = dict()

    # Initialize net
    big_net = AllInOneNet(configuration)
    big_net.load_state_dict(model["big_net_state_dict"])

    with torch.no_grad():
        for degree in range(1, 1 + configuration["total_degrees"]):
            evaluation_performance[degree] = list()

            couples = objectives_to_coupled_proofs_and_provers(
                encoder=big_net.encoder,
                objectives=pickle.load(open(configuration["objective_path"].format(degree, theorems_per_degree), "rb")),
                simple_proof=pickle.load(open(configuration["simple_proof_path"], "rb"))
            )

            single_degree_performance = list()
            for proof, prover in couples:
                for step in range(2 * configuration["total_degrees"]):

                    # Get action spaces and their representations
                    lower_action_space = list(proof.entities)
                    lower_action_tensor_space = big_net.encoder.encode_entity_space(lower_action_space)
                    higher_action_space = proof.lemmas
                    higher_action_tensor_space = big_net.encoder.encode_theorem_space(higher_action_space)

                    # Make higher decision
                    higher_state = prover.raw_observe()
                    higher_state_tensor = big_net.encoder.higher_attentive_state(higher_state)
                    theorem_chosen = \
                        big_net.higher_net.sample_action(state=higher_state_tensor, action_space=higher_action_space,
                                                         action_tensor_space=higher_action_tensor_space)[0][0]
                    del higher_state_tensor
                    del higher_action_tensor_space

                    # Make lower decisions
                    sequence_of_operands = list()
                    for operand_index in range(1, 1 + theorem_chosen.input_no):
                        lower_state = (prover.raw_observe(), theorem_chosen, sequence_of_operands)
                        lower_state_tensor = big_net.encoder.lower_attentive_state(observation=lower_state[0],
                                                                                   lemma_chosen=lower_state[1],
                                                                                   sequence_of_entities=lower_state[2])
                        entity_chosen = \
                            big_net.lower_net.sample_action(state=lower_state_tensor,
                                                            action_space=lower_action_space,
                                                            action_tensor_space=lower_action_tensor_space)[0][0]
                        sequence_of_operands.append(entity_chosen)
                        lower_reward, reward_string, conclusions = \
                            proof.apply_theorem_get_conclusions_and_reward(
                                theorem=theorem_chosen,
                                operands=sequence_of_operands,
                                reward_scheme=configuration["lower_reward_scheme"]
                            )
                        proof.trim_entities(max_number_of_entities=configuration["ent_maxsize"])
                        proof.trim_ground_truth(max_number_of_ground_truth=configuration["gt_maxsize"])

                    if proof.proved:
                        break
                if proof.proved:
                    single_degree_performance.append(1)
                else:
                    single_degree_performance.append(0)
            evaluation_performance[degree].append(sum(single_degree_performance) / len(couples))
    json.dump(evaluation_performance, open(configuration["saving_dir"] + "evaluation_performance.json", "w"))


if __name__ == "__main__":
    configuration = load_config()["naive_her"]
    evaluate(configuration)
