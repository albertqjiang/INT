import os
import sys
import json
from copy import deepcopy

import torch

sys.path.insert(0, os.path.abspath('../..'))
from baselines.supervised import initialize
from logic.logic import Proof


def lower_baseline(imitation_net, val, number_of_trials=10):
    """
    This is the real supervised learning baseline. It tries to prove problems and calculate the fraction of problems
    solved instead of the 'accuracy'. It is directly comparable with reinforcement learning agents. It doesn't have the
    problem of having to choose one in a one-to-many mapping.
    :param imitation_net: the main imitation learning network trained
    :param val: validation dataset
    :param number_of_trials: number of trials allowed to get to the final objective
    :return: fraction of problems solved, and all the cases the agent not able to solve
    """
    proofs = list()
    with torch.no_grad():
        base_proof = Proof(entities=list(), axioms=list(), assumptions=list(), objectives=list())

        success = 0
        wrong_cases = list()
        for trajectory in val:
            single_proof = dict()
            single_proof["proof_steps"] = list()

            custom_proof = deepcopy(base_proof)
            custom_proof.proved = False
            custom_proof.entities = trajectory["observation"]["entities"]
            custom_proof.lemmas = trajectory["observation"]["lemmas"]
            custom_proof.ground_truth = trajectory["observation"]["ground_truth"]
            custom_proof.objectives = trajectory["observation"]["objectives"]

            single_proof["objective"] = custom_proof.objectives[0].name

            trial = 0
            previous_steps = list()
            while (trial < number_of_trials) and (not custom_proof.proved):
                # Trim the previous steps if they are too long
                # This is done during training so should also be done during evaluation
                if len(previous_steps) > 3:
                    previous_steps = previous_steps[-3:]

                current_observation = custom_proof.get_observation()
                lemma_chosen, _ = imitation_net.higher_sample_action(observation=current_observation)
                lemma_chosen = lemma_chosen[0]
                sequence_of_entities = list()
                for i in range(lemma_chosen.input_no):
                    entity_chosen, _ = imitation_net.lower_sample_action(
                        observation=current_observation,
                        lemma=lemma_chosen,
                        sequence_of_entities=sequence_of_entities,
                        previous_steps=previous_steps,
                        entity_space=current_observation["entities"],
                    )

                    sequence_of_entities.extend(entity_chosen)

                previous_steps.append({
                    "observation": custom_proof.get_observation(),
                    "lemma": lemma_chosen,
                    "input_entities": sequence_of_entities
                })
                custom_proof.apply_theorem(theorem=lemma_chosen, operands=sequence_of_entities)

                trial += 1
                single_proof["proof_steps"].append({"lemma": lemma_chosen.name,
                                                    "operands": [operand.name for operand in sequence_of_entities],
                                                    "proved": str(custom_proof.proved)})

            if custom_proof.proved:
                success += 1
            else:
                wrong_cases.append(previous_steps)

            proofs.append(single_proof)
    json.dump(proofs, open("../pt_models/supervised_baseline/validation_proofs.json", "w"))
    return success / len(val), wrong_cases


if __name__ == "__main__":
    trajectories, batch_size, train_val_split, criterion, epochs, imitation_net, optimizer, update, \
    lower_training_losses, higher_training_losses = initialize(custom_objective_dir="../data/test_trajectories.p",
                                                               resume=True, cuda=False)
    try:
        val_accuracies = json.load(open("../pt_models/supervised_baseline/val_accuracies.json", "r"))
    except Exception:
        val_accuracies = list()
    val_set = trajectories[int(train_val_split * len(trajectories)):]
    success_rate, wrong_cases = lower_baseline(imitation_net, val_set)
    val_accuracies.append(success_rate)
    json.dump(val_accuracies, open("../pt_models/supervised_baseline/val_accuracies.json", "w"))
