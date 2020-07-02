import os
import sys

sys.path.insert(0, os.path.abspath('../../..'))
from legacy.rl_algos.multi_step import objectives_to_coupled_proofs_and_provers
from copy import deepcopy

import torch
import pickle


def test_agent(degree, configuration, agent_path, objectives_path):
    higher_rewards = list()
    lower_rewards = list()
    optimality = list()
    proved = list()
    list_of_proofs = list()

    simple_proof_path = configuration["simple_proof_path"]
    max_steps = configuration["max_step"]
    lower_reward_scheme = configuration["lower_reward_scheme"]
    higher_reward_scheme = configuration["higher_reward_scheme"]

    big_net = torch.load(agent_path)["big_net"]
    objectives = pickle.load(open(objectives_path, "rb"))
    simple_proof = pickle.load(open(simple_proof_path, "rb"))
    list_of_proof_and_provers = objectives_to_coupled_proofs_and_provers(objectives=objectives,
                                                                         encoder=big_net.encoder,
                                                                         simple_proof=simple_proof)
    this_encoder = big_net.encoder
    lower_q_net = big_net.lower_net
    higher_q_net = big_net.higher_net

    for proof, prover in list_of_proof_and_provers:
        this_proof, this_prover = deepcopy(proof), deepcopy(prover)
        total_higher_reward = 0
        total_lower_reward = 0
        lower_step = 0
        for outer_step in range(1, 1 + max_steps):
            if (len(this_proof.ground_truth) > this_encoder.ground_truth_maxsize - 2) or \
                    (len(this_proof.lemmas) > this_encoder.theorem_maxsize - 1) or \
                    (len(this_proof.entities) > this_encoder.entity_maxsize - 5):
                outer_step = max_steps
                break
            list_of_entities = list(this_proof.entities)
            lower_action_space = list_of_entities
            higher_action_space = this_proof.lemmas
            lower_action_tensor_space = torch.cat(
                [this_encoder.encode_entity(entity) for entity in list_of_entities],
                dim=0)
            higher_action_tensor_space = torch.cat(
                [this_encoder.encode_theorem(lemma) for lemma in this_proof.lemmas],
                dim=0)

            higher_state = this_encoder.higher_attentive_state(this_prover.raw_observe())
            lemma_chosen = higher_q_net.sample_action(state=higher_state, action_space=higher_action_space,
                                                      action_tensor_space=higher_action_tensor_space)[0]
            sequence_of_operands = list()
            for operand_index in range(1, 1 + lemma_chosen.input_no):
                # Lower MDP and update
                lower_state = this_encoder.lower_attentive_state(observation=this_prover.raw_observe(),
                                                                 lemma_chosen=lemma_chosen,
                                                                 sequence_of_entities=sequence_of_operands)
                entity_chosen = lower_q_net.sample_action(state=lower_state, action_space=lower_action_space,
                                                          action_tensor_space=lower_action_tensor_space)[0]
                sequence_of_operands.append(entity_chosen)
                lower_reward, reward_string = this_proof.apply_theorem_get_reward(theorem=lemma_chosen,
                                                                                  operands=sequence_of_operands,
                                                                                  reward_scheme=lower_reward_scheme)
                total_lower_reward += lower_reward
                lower_step += 1

            # Higher MDP and update
            higher_reward = higher_reward_scheme[reward_string]
            total_higher_reward += higher_reward

            if this_proof.proved:
                break

        proved.append(int(this_proof.proved))
        higher_rewards.append(total_higher_reward / outer_step)
        lower_rewards.append(total_lower_reward / (lower_step - 0.001))
        optimality.append(outer_step / degree)
        if this_proof.proved:
            list_of_proofs.append(this_proof.print_proof_status())
        else:
            list_of_proofs.append("Didn't manage to prove:\n")
            list_of_proofs.append(this_proof.print_proof_status())
    return higher_rewards, lower_rewards, optimality, list_of_proofs, proved


if __name__ == "__main__":
    from legacy.helper_functions.utils import load_config
    import json

    config = load_config()["multi_step_q"]
    degree = 1
    testing_results = \
        test_agent(
            degree=degree, configuration=config,
            agent_path="../../pt_models/imitation_aided/model.pt",
            objectives_path="../../data/objectives_by_degree/simple_proof_degree_{}_100_objectives.p".format(degree))
    testing_results_dict = {name: result for name, result in zip(("hr", "lr", "op", "lp", "pr"), testing_results)}
    json.dump(testing_results_dict, open("testing_results.json", "w"))
