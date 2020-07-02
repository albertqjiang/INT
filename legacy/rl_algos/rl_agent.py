import json
import os
import sys
import pickle
import random
import gc

from copy import deepcopy
from operator import itemgetter

import torch
import torch.optim as optim
import torch.nn as nn

sys.path.insert(0, os.path.abspath('../../../..'))

from legacy.helper_functions.utils import load_config, set_random_seed, ReplayBuffer, normalize_rewards
from legacy.helper_functions.QNetwork import GeneralQNetwork
from logic.logic import Proof
from legacy.logic_math import real_number_axioms
from legacy.helper_functions.BatchEncoder import BatchEncoder


class AllInOneNet(nn.Module):
    def __init__(self, configuration, encoder_configuration, cuda=True):
        super(AllInOneNet, self).__init__()
        self.configuration = configuration

        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.encoder = BatchEncoder(configuration=encoder_configuration, cuda=cuda)

        # Determine particular training scheme and relevant dimensions
        self.higher_state_dim = self.encoder.higher_attentive_size
        self.higher_action_dim = self.encoder.theorem_embedding_size
        self.higher_hidden_dim = self.configuration["higher_hidden_dim"]
        self.lower_state_dim = self.encoder.lower_attentive_size
        self.lower_action_dim = self.encoder.entity_embedding_size
        self.lower_hidden_dim = self.configuration["lower_hidden_dim"]

        # Initialize higher and lower neural nets
        self.higher_net = GeneralQNetwork(
            state_dim=self.higher_state_dim, action_dim=self.higher_action_dim, hidden_dim=self.higher_hidden_dim,
            cuda=cuda
        )
        self.lower_net = GeneralQNetwork(
            state_dim=self.lower_state_dim, action_dim=self.lower_action_dim, hidden_dim=self.lower_hidden_dim,
            cuda=cuda
        )

    def forward(self, x):
        raise NotImplementedError


def prepare_proofs(objectives, simple_proof, assumptions=[]):
    proofs = list()
    for objective in objectives:
        entity_strings = {entity.name: True for entity in simple_proof.initial_entities}
        all_entities = simple_proof.initial_entities

        # Augment the entities with all entities that appeared in the logic statements
        trace_list = list()
        for logic_statement in assumptions + [objective]:
            trace_list.extend([entity for entity in logic_statement.entities])
        while trace_list:
            entity_to_explore = trace_list.pop()
            if not entity_to_explore.recent_numerical_function:
                pass
            else:
                if entity_to_explore.name in entity_strings:
                    pass
                else:
                    all_entities.append(entity_to_explore)
                    trace_list.extend(entity_to_explore.rnc_operands)

        proof = Proof(entities=all_entities, assumptions=assumptions,
                      axioms=[axiom for axiom in real_number_axioms.values()], objectives=[objective])
        proofs.append(proof)
    return proofs


def update_lower(lower_net, optimizer, encoder, entity_space, buffer, configuration, epochs=1, update=0):
    batch_size = configuration["batch_size"]
    discount_factor = configuration["discount_factor"]
    td_errors = list()
    for epoch in range(epochs):
        batch = buffer.sample(batch_size)
        batch_state = batch["state"]
        batch_action = batch["action"]
        batch_next_state = batch["next_state"]
        batch_reward = batch["reward"]
        batch_done = batch["done"]

        state_tensor = torch.cat([encoder.lower_attentive_state(
            observation=state[0], lemma_chosen=state[1], sequence_of_entities=state[2])
            for state in batch_state], dim=0)
        action_tensor = encoder.batch_encode(batch=batch_action, value_type="entity")
        next_state_tensor = torch.cat([encoder.lower_attentive_state(
            observation=next_state[0], lemma_chosen=next_state[1], sequence_of_entities=next_state[2])
            for next_state in batch_next_state], dim=0)
        reward_tensor = torch.cat(batch_reward, dim=0)
        reward_tensor = \
            normalize_rewards(reward_tensor,
                              normalization_value=configuration["lower_reward_scheme"]["REWARD_PROOF_COMPLETE"])
        done_tensor = torch.cat(batch_done, dim=0)
        action_tensor_space = encoder.encode_entity_space(entity_space)

        loss = lower_net.q_loss(state=state_tensor, action=action_tensor, next_state=next_state_tensor,
                                reward=reward_tensor, done=done_tensor,
                                action_tensor_space=action_tensor_space, discount_factor=discount_factor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        td_errors.append(loss.item())
        update += 1

        # Release memory
        del state_tensor, action_tensor, next_state_tensor, reward_tensor, done_tensor, action_tensor_space, loss
        torch.cuda.empty_cache()
        gc.collect()
    return td_errors, update


def update_higher(higher_net, optimizer, encoder, theorem_space, buffer, configuration, epochs=1, update=0):
    batch_size = configuration["batch_size"]
    discount_factor = configuration["discount_factor"]
    td_errors = list()
    for epoch in range(epochs):
        batch = buffer.sample(batch_size)
        batch_state = batch["state"]
        batch_action = batch["action"]
        batch_next_state = batch["next_state"]
        batch_reward = batch["reward"]
        batch_done = batch["done"]

        state_tensor = torch.cat([encoder.higher_attentive_state(observation=state)
                                  for state in batch_state], dim=0)
        action_tensor = encoder.batch_encode(batch=batch_action, value_type="theorem")
        next_state_tensor = torch.cat([encoder.higher_attentive_state(observation=state)
                                       for state in batch_next_state], dim=0)
        reward_tensor = torch.cat(batch_reward, dim=0)
        reward_tensor = \
            normalize_rewards(reward_tensor,
                              normalization_value=configuration["higher_reward_scheme"]["REWARD_PROOF_COMPLETE"])
        done_tensor = torch.cat(batch_done, dim=0)
        action_tensor_space = encoder.encode_theorem_space(theorem_space)
        loss = higher_net.q_loss(state=state_tensor, action=action_tensor, reward=reward_tensor,
                                 next_state=next_state_tensor, done=done_tensor,
                                 action_tensor_space=action_tensor_space,
                                 discount_factor=discount_factor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        td_errors.append(loss.item())
        update += 1

        # Release memory
        del state_tensor, action_tensor, next_state_tensor, reward_tensor, done_tensor, action_tensor_space, loss
        torch.cuda.empty_cache()
        gc.collect()
    return td_errors, update


def initialize_training(configuration, encoder_configuration, resume=False, pretrained=False):
    # Get info from configuration
    config_keys = ["saving_dir", "lower_reward_scheme", "higher_reward_scheme", "batch_size",
                   "total_updates_per_degree",
                   "objective_path", "simple_proof_path", "objectives_per_degree", "higher_epochs", "lower_epochs",
                   "ent_maxsize", "gt_maxsize"]
    saving_directory, lower_reward_scheme, higher_reward_scheme, batch_size, total_updates_per_degree, \
    objective_path, simple_proof_path, objectives_per_degree, higher_epochs, lower_epochs, \
    max_number_of_entities, max_number_of_ground_truth = itemgetter(*config_keys)(configuration)

    # Initialize net
    big_net = AllInOneNet(configuration, encoder_configuration)

    # Initialize optimizers
    higher_optimizer = optim.Adam(
        params=big_net.higher_net.parameters(), lr=configuration["higher_lr"], amsgrad=False)
    lower_encoder_optimizer = optim.Adam(
        params=list(big_net.lower_net.parameters()) + list(big_net.encoder.parameters()),
        lr=configuration["lower_lr"], amsgrad=False)

    if not resume:
        # Initialize training info record
        training_info = {
            "lower_rewards": list(),
            "higher_rewards": list(),
            "proved": list(),
            "updates": list(),
            "time": list(),
            "lower td error": list(),
            "higher td error": list()
        }
        proof_info = list()

        # Initialize buffers
        higher_buffer = ReplayBuffer(name="higher_buffer", capacity=configuration["higher_buffer_capacity"])
        lower_buffer = ReplayBuffer(name="lower_buffer", capacity=configuration["lower_buffer_capacity"])

        # Use the pretrained encoder and copy stuff to gpu to initialize it
        if pretrained:
            model = torch.load(configuration["supervised_model_dir"])
            big_net.encoder.load_state_dict(model['encoder_state_dict'])
            big_net.encoder.to(big_net.device)
            del model
            gc.collect()
    else:
        model = torch.load(saving_directory + "model.pt")
        big_net.load_state_dict(model["big_net_state_dict"])
        higher_optimizer.load_state_dict(model["higher_q_optimizer_state_dict"])
        lower_encoder_optimizer.load_state_dict(model["lower_encoder_q_optimizer_state_dict"])
        training_info = model["training_info"]
        proof_info = model["proof_info"]
        lower_buffer = model["lower_buffer"]
        higher_buffer = model["higher_buffer"]

    return saving_directory, lower_reward_scheme, higher_reward_scheme, batch_size, total_updates_per_degree, \
           objective_path, simple_proof_path, objectives_per_degree, higher_epochs, lower_epochs, \
           max_number_of_entities, max_number_of_ground_truth, \
           training_info, proof_info, \
           big_net, higher_optimizer, lower_encoder_optimizer, higher_buffer, lower_buffer


def train_on_seed(seed, configuration, encoder_configuration, her=True, resume=False, pretrained=False):
    set_random_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    saving_directory, lower_reward_scheme, higher_reward_scheme, batch_size, total_updates_per_degree, \
    objective_path, simple_proof_path, objectives_per_degree, higher_epochs, lower_epochs, \
    max_number_of_entities, max_number_of_ground_truth, \
    training_info, proof_info, \
    big_net, higher_optimizer, lower_encoder_optimizer, higher_buffer, lower_buffer \
        = initialize_training(configuration,
                              encoder_configuration=encoder_configuration, resume=resume, pretrained=pretrained)

    torch.cuda.empty_cache()
    training_trigger = False

    all_proofs = list()

    # Get objectives and proof-prover couples
    for i in range(configuration["number_of_configurations"]):
        simple_proof = pickle.load(open(simple_proof_path, "rb"))
        if os.path.getsize("../../data/standard_theorem_dataset/assumptions_{}.p".format(i)) > 0:
            assumptions = pickle.load(open("../../data/standard_theorem_dataset/assumptions_{}.p".format(i), "rb"))
        else:
            assumptions = list()

        proof_wrapping_objectives = pickle.load(open("../../data/standard_theorem_dataset/proof_{}.p".format(i), "rb"))
        objectives = proof_wrapping_objectives.ground_truth

        proofs = prepare_proofs(objectives=objectives, simple_proof=simple_proof, assumptions=assumptions)
        all_proofs.extend(proofs)

    update = 0
    total_proof_steps = list()
    while update < total_updates_per_degree:
        # Initialize record variables
        aggregate_lower_reward = 0
        aggregate_lower_steps = 0
        aggregate_higher_reward = 0
        aggregate_higher_steps = 0

        proof = deepcopy(random.choice(all_proofs))
        # This is to prevent the situation where there are theorems with more than one objective generated
        if len(proof.objectives) > 1:
            print("Found proof with more than one objective")
            continue
        single_proof_steps = list()

        if her:
            lower_her_transitions = list()
            higher_her_transitions = list()
        for step in range(configuration["max_step"]):
            with torch.no_grad():
                # Get action spaces and their representations
                lower_action_space = list(proof.entities)
                lower_action_tensor_space = big_net.encoder.encode_entity_space(lower_action_space)
                higher_action_space = proof.lemmas
                higher_action_tensor_space = big_net.encoder.encode_theorem_space(higher_action_space)

                # Make higher decision
                higher_state = proof.get_observation()
                higher_state_tensor = big_net.encoder.higher_attentive_state(higher_state)
                theorem_chosen = \
                    big_net.higher_net.sample_action(state=higher_state_tensor, action_space=higher_action_space,
                                                     action_tensor_space=higher_action_tensor_space)[0][0]
                single_proof_steps.append([theorem_chosen.name])
                del higher_state_tensor
                del higher_action_tensor_space

                # Make lower decisions
                sequence_of_operands = list()
                for operand_index in range(1, 1 + theorem_chosen.input_no):
                    lower_state = (proof.get_observation(), theorem_chosen, sequence_of_operands)
                    lower_state_tensor = big_net.encoder.lower_attentive_state(observation=lower_state[0],
                                                                               lemma_chosen=lower_state[1],
                                                                               sequence_of_entities=lower_state[2])
                    entity_chosen = big_net.lower_net.sample_action(state=lower_state_tensor,
                                                                    action_space=lower_action_space,
                                                                    action_tensor_space=lower_action_tensor_space)[0][0]
                    sequence_of_operands.append(entity_chosen)
                    lower_next_state = (proof.get_observation(), theorem_chosen, sequence_of_operands)
                    lower_reward, reward_string, conclusions = proof.apply_theorem_get_conclusions_and_reward(
                        theorem=theorem_chosen, operands=sequence_of_operands, reward_scheme=lower_reward_scheme
                    )
                    proof.trim_entities(max_number_of_entities=max_number_of_entities)
                    proof.trim_ground_truth(max_number_of_ground_truth=max_number_of_ground_truth)
                    aggregate_lower_reward += lower_reward
                    aggregate_lower_steps += 1
                    lower_reward = torch.FloatTensor([[lower_reward]]).to(device)
                    lower_done = 1 if (operand_index == theorem_chosen.input_no) else 0
                    lower_done = torch.FloatTensor([[lower_done]]).to(device)
                    lower_buffer.push(state=lower_state, next_state=lower_next_state, reward=lower_reward,
                                      action=entity_chosen, done=lower_done)
                    if her:
                        lower_her_transitions.append({"state": lower_state, "next_state": lower_next_state,
                                                      "reward": lower_reward, "action": entity_chosen,
                                                      "done": lower_done})

                    single_proof_steps[-1].append(entity_chosen.name)

                # Higher MDP and update
                higher_next_state = proof.get_observation()
                higher_done = 1 if proof.proved else 0
                higher_done = torch.FloatTensor([[higher_done]]).to(device)
                higher_reward = higher_reward_scheme[reward_string]
                aggregate_higher_reward += higher_reward
                aggregate_higher_steps += 1
                higher_reward = torch.FloatTensor([[higher_reward]]).to(device)
                higher_buffer.push(state=higher_state, next_state=higher_next_state, reward=higher_reward,
                                   action=theorem_chosen, done=higher_done)
                if her:
                    higher_her_transitions.append({"state": higher_state, "next_state": higher_next_state,
                                                   "reward": higher_reward, "action": theorem_chosen,
                                                   "done": higher_done})

                # HER: if a new statement is proved, put all transitions in higher and lower her transitions into the
                # main replay buffer with the modified objective being the newly proved statement
                if reward_string == "REWARD_THEOREM_PROCEEDED" and her:
                    for conclusion in conclusions:
                        for i in range(len(higher_her_transitions)):
                            higher_her_transitions[i]["state"]["objectives"] = [conclusion]
                            higher_her_transitions[i]["next_state"]["objectives"] = [conclusion]
                        higher_her_transitions[-1]["done"] = torch.FloatTensor([[1]]).to(device)
                        higher_her_transitions[-1]["reward"] = \
                            torch.FloatTensor([[higher_reward_scheme["REWARD_PROOF_COMPLETE"]]]).to(device)

                        for j in range(len(lower_her_transitions)):
                            lower_her_transitions[j]["state"][0]["objectives"] = [conclusion]
                            lower_her_transitions[j]["next_state"][0]["objectives"] = [conclusion]
                        lower_her_transitions[-1]["done"] = torch.FloatTensor([[1]]).to(device)
                        lower_her_transitions[-1]["reward"] = \
                            torch.FloatTensor([[lower_reward_scheme["REWARD_PROOF_COMPLETE"]]]).to(device)

                        for hh_transition in higher_her_transitions:
                            higher_buffer.push(state=hh_transition["state"], next_state=hh_transition["next_state"],
                                               reward=hh_transition["reward"],
                                               action=hh_transition["action"], done=hh_transition["done"])

                        for lh_transition in lower_her_transitions:
                            lower_buffer.push(state=lh_transition["state"], next_state=lh_transition["next_state"],
                                              reward=lh_transition["reward"],
                                              action=lh_transition["action"], done=lh_transition["done"])

                    training_trigger = True

                if proof.proved:
                    break

        training_info["lower_rewards"].append(aggregate_lower_reward / aggregate_lower_steps)
        training_info["higher_rewards"].append(aggregate_higher_reward / aggregate_higher_steps)
        training_info["proved"].append(int(proof.proved))
        training_info["updates"].append(update)

        if len(lower_buffer) >= batch_size and (her or training_trigger):
            lower_td_errors, update = \
                update_lower(lower_net=big_net.lower_net, buffer=lower_buffer, configuration=configuration,
                             encoder=big_net.encoder, entity_space=proof.entities,
                             optimizer=lower_encoder_optimizer, epochs=lower_epochs, update=update)
        if len(higher_buffer) >= batch_size and (her or training_trigger):
            higher_td_errors, update = \
                update_higher(higher_net=big_net.higher_net, buffer=higher_buffer, configuration=configuration,
                              encoder=big_net.encoder, optimizer=higher_optimizer, theorem_space=proof.lemmas,
                              epochs=higher_epochs, update=update)

        # Update training info and checkpoint
        if len(lower_buffer) >= batch_size and len(higher_buffer) >= batch_size and \
                len(higher_buffer) >= batch_size and (her or training_trigger):
            training_info["lower td error"].extend(lower_td_errors)
            training_info["higher td error"].extend(higher_td_errors)
            if pretrained:
                json.dump(training_info, open(saving_directory + "training_info_pretrained.json", "w"))
            else:
                json.dump(training_info, open(saving_directory + "training_info.json", "w"))

        total_proof_steps.append(single_proof_steps)
        model = {
            "big_net_state_dict": big_net.state_dict(),
            "higher_q_optimizer_state_dict": higher_optimizer.state_dict(),
            "lower_encoder_q_optimizer_state_dict": lower_encoder_optimizer.state_dict(),
            "training_info": training_info, "total_proof_steps": total_proof_steps, "proof_info": proof_info,
            "lower_buffer": lower_buffer, "higher_buffer": higher_buffer,
        }
        torch.save(model, saving_directory + "model.pt")


def run(configuration, encoder_configuration, resume=False, her=True, pretrained=False):
    random_seeds = configuration["total_random_seeds"]
    for random_seed in range(1, 1 + random_seeds):
        train_on_seed(random_seed, configuration,
                      encoder_configuration=encoder_configuration, her=her, resume=resume, pretrained=pretrained)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--resume", required=False, default="False",
                    help="whether to resume training")
    ap.add_argument("-p", "--pretrained", required=False, default="False",
                    help="whether to initialize network with supervised encoder")
    ap.add_argument("-her", "--her", required=False, default="True",
                    help="whether to use hindsight experience replay")
    args = vars(ap.parse_args())

    resume_string = args['resume']
    if resume_string.startswith(('t', 'T')):
        resume = True
        print("Resume training.")
    elif resume_string.startswith(('f', "F")):
        resume = False
        print("Start training.")
    else:
        raise NotImplementedError

    pretrained_string = args['pretrained']
    if pretrained_string.startswith(('t', 'T')):
        pretrained = True
        print("Used pretrained encoder.")
    elif pretrained_string.startswith(('f', "F")):
        pretrained = False
        print("Training end-to-end.")
    else:
        raise NotImplementedError

    her_string = args['her']
    if her_string.startswith(('t', 'T')):
        her = True
        print("Using Hindsight Experience Replay.")
    elif her_string.startswith(('f', "F")):
        her = False
        print("NOT using Hindsight Experience Replay.")
    else:
        raise NotImplementedError

    config = load_config()["supervised_her"]
    encoder_configuration = load_config()["batch_encode"]
    run(config, resume=resume, her=her, pretrained=pretrained, encoder_configuration=encoder_configuration)
