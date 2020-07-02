import json
import os
import sys
import time
import pickle
import random
import torch
import torch.optim as optim
from copy import deepcopy

sys.path.insert(0, os.path.abspath('../../..'))

from legacy.helper_functions.utils import load_config, set_random_seed, ReplayBuffer, prepare_batches, \
    convert_batch_dict_to_dict_batch
from legacy.helper_functions.QNetwork import GeneralQNetwork
from legacy.helper_functions.RecursiveAttentiveEncoder import RecursiveAttentiveEncoder
from legacy.logic.logicRL import LogicBasedProver


def proof_templates_to_coupled_proofs_and_provers(proof_templates, encoder):
    proof_prover_couples = list()
    for proof_template in proof_templates:
        couple_prover = LogicBasedProver(
            proof=proof_template, ent_maxsize=encoder.entity_maxsize, gt_maxsize=encoder.ground_truth_maxsize,
            lemma_maxsize=encoder.theorem_maxsize, objective_maxsize=encoder.objective_maxsize,
            lemma_operand_size=encoder.theorem_operand_maxsize, lemma_embedding_size=encoder.theorem_embedding_size,
        )
        proof_prover_couples.append((proof_template, couple_prover))
    return proof_prover_couples


def update_lower(lower_net, optimizer, encoder, entity_space, buffer, configuration, epochs=1):
    representation_time, update_time = 0, 0
    batch_size = configuration["batch_size"]
    discount_factor = configuration["discount_factor"]
    td_errors = list()
    for epoch in range(epochs):
        list_of_batches = prepare_batches(buffer, batch_size)
        for batch in list_of_batches:
            dict_batch = convert_batch_dict_to_dict_batch(batch)
            batch_state = dict_batch["state"]
            batch_action = dict_batch["action"]
            batch_next_state = dict_batch["next_state"]
            batch_reward = dict_batch["reward"]
            batch_done = dict_batch["done"]

            starting_time = time.time()
            state_tensor = torch.cat([encoder.lower_attentive_state(
                observation=state[0], lemma_chosen=state[1], sequence_of_entities=state[2])
                for state in batch_state], dim=0)
            action_tensor = torch.cat([encoder.encode_entity(action) for action in batch_action], dim=0)
            next_state_tensor = torch.cat([encoder.lower_attentive_state(
                observation=next_state[0], lemma_chosen=next_state[1], sequence_of_entities=next_state[2])
                for next_state in batch_next_state], dim=0)
            reward_tensor = torch.cat(batch_reward, dim=0)
            done_tensor = torch.cat(batch_done, dim=0)
            action_tensor_space = encoder.encode_entity_space(entity_space)
            representation_time += time.time() - starting_time
            starting_time = time.time()

            loss = lower_net.q_loss(state=state_tensor, action=action_tensor, next_state=next_state_tensor,
                                    reward=reward_tensor, done=done_tensor,
                                    action_tensor_space=action_tensor_space, discount_factor=discount_factor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_time += time.time() - starting_time
            td_errors.append(loss.item())
    return representation_time, update_time, td_errors


def update_higher(higher_net, optimizer, encoder, theorem_space, buffer, configuration, epochs=1):
    representation_time, update_time = 0, 0
    batch_size = configuration["batch_size"]
    discount_factor = configuration["discount_factor"]
    td_errors = list()
    for epoch in range(epochs):
        list_of_batches = prepare_batches(buffer, batch_size)
        for batch in list_of_batches:
            dict_batch = convert_batch_dict_to_dict_batch(batch)
            batch_state = dict_batch["state"]
            batch_action = dict_batch["action"]
            batch_next_state = dict_batch["next_state"]
            batch_reward = dict_batch["reward"]
            batch_done = dict_batch["done"]

            starting_time = time.time()
            state_tensor = torch.cat([encoder.higher_attentive_state(observation=state)
                                      for state in batch_state], dim=0)
            action_tensor = torch.cat([encoder.encode_theorem(action) for action in batch_action], dim=0)
            next_state_tensor = torch.cat([encoder.higher_attentive_state(observation=state)
                                           for state in batch_next_state], dim=0)
            reward_tensor = torch.cat(batch_reward, dim=0)
            done_tensor = torch.cat(batch_done, dim=0)
            action_tensor_space = encoder.encode_theorem_space(theorem_space)
            representation_time += time.time() - starting_time
            starting_time = time.time()
            loss = higher_net.q_loss(state=state_tensor, action=action_tensor, reward=reward_tensor,
                                     next_state=next_state_tensor, done=done_tensor,
                                     action_tensor_space=action_tensor_space,
                                     discount_factor=discount_factor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_time += time.time() - starting_time
            td_errors.append(loss.item())
    return representation_time, update_time, td_errors


class AllInOneNet:
    def __init__(self, configuration):

        self.configuration = configuration

        # Initialize encoder
        simple_proof = pickle.load(open(self.configuration["simple_proof_path"], "rb"))
        if self.configuration["pretrain"] == "False":
            self.encoder = RecursiveAttentiveEncoder(simple_proof, self.configuration)
        elif "encoder_path" in self.configuration.keys():
            self.encoder = torch.load(self.configuration["encoder_path"])
        else:
            raise NotImplementedError

        # Determine particular training scheme and relevant dimensions
        self.training_scheme = self.configuration["training_scheme"]
        if "attentive" in self.training_scheme:
            self.higher_state_dim = self.encoder.higher_attentive_size
            self.higher_action_dim = self.encoder.theorem_embedding_size
            self.higher_hidden_dim = self.configuration["higher_hidden_dim"]
            self.lower_state_dim = self.encoder.lower_attentive_size
            self.lower_action_dim = self.encoder.entity_embedding_size
            self.lower_hidden_dim = self.configuration["lower_hidden_dim"]
        else:
            raise NotImplementedError

        # Initialize higher and lower neural nets
        self.higher_net = GeneralQNetwork(
            state_dim=self.higher_state_dim, action_dim=self.higher_action_dim, hidden_dim=self.higher_hidden_dim
        )
        self.lower_net = GeneralQNetwork(
            state_dim=self.lower_state_dim, action_dim=self.lower_action_dim, hidden_dim=self.lower_hidden_dim
        )


def run(configuration):
    random_seeds = configuration["total_random_seeds"]
    for random_seed in range(1, 1 + random_seeds):
        train_on_seed(random_seed, configuration)


def train_on_seed(seed, configuration):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(seed)

    # Get info from configuration
    saving_directory = configuration["saving_dir"]
    lower_reward_scheme = configuration["lower_reward_scheme"]
    higher_reward_scheme = configuration["higher_reward_scheme"]
    batch_size = configuration["batch_size"]
    total_updates_per_degree = configuration["total_updates_per_degree"]
    proof_template_path = configuration["proof_template_path"]
    higher_epochs = configuration["higher_epochs"]
    lower_epochs = configuration["lower_epochs"]
    imitation_reward = configuration["imitation_reward"]
    max_number_of_entities = configuration["ent_maxsize"]
    max_number_of_ground_truth = configuration["gt_maxsize"]

    # Initialize training info record
    proof_info = list()

    # Initialize net
    big_net = AllInOneNet(configuration)

    # Initialize optimizers
    higher_optimizer = optim.Adam(
        params=big_net.higher_net.parameters(), lr=configuration["higher_lr"], amsgrad=False)
    lower_encoder_optimizer = optim.Adam(
        params=list(big_net.lower_net.parameters()) + list(big_net.encoder.parameters()),
        lr=configuration["lower_lr"], amsgrad=False)

    # Initialize buffers
    higher_buffer = ReplayBuffer(name="higher_buffer", capacity=configuration["higher_buffer_capacity"])
    higher_success_buffer = ReplayBuffer(name="success_buffer", capacity=configuration["success_buffer_capacity"])
    lower_buffer = ReplayBuffer(name="lower_buffer", capacity=configuration["lower_buffer_capacity"])
    lower_success_buffer = ReplayBuffer(name="success_buffer", capacity=configuration["success_buffer_capacity"])

    training_trigger = False
    training_info = {
        "lower_rewards": list(),
        "higher_rewards": list(),
        "optimality": list(),
        "proved": list(),
        "updates": list(),
        "time": list(),
        "buffer size": list(),
        "lower td error": list(),
        "higher td error": list()
    }

    # Get objectives and proof-prover couples
    couples = proof_templates_to_coupled_proofs_and_provers(
        encoder=big_net.encoder,
        proof_templates=pickle.load(open(proof_template_path, "rb")),
    )

    lower_episode_list = list()
    higher_episode_list = list()
    update = 0
    total_proof_steps = list()
    while update < total_updates_per_degree:
        # Initialize record variables
        aggregate_lower_reward = 0
        aggregate_lower_steps = 0
        aggregate_higher_reward = 0
        aggregate_higher_steps = 0

        proof, prover = deepcopy(random.choice(couples))
        single_proof_steps = list()
        for step in range(configuration["max_step"]):
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
                                                 action_tensor_space=higher_action_tensor_space)[0]
            single_proof_steps.append([theorem_chosen.name])

            # Make lower decisions
            sequence_of_operands = list()
            for operand_index in range(1, 1 + theorem_chosen.input_no):
                lower_state = (prover.raw_observe(), theorem_chosen, sequence_of_operands)
                lower_state_tensor = big_net.encoder.lower_attentive_state(observation=lower_state[0],
                                                                           lemma_chosen=lower_state[1],
                                                                           sequence_of_entities=lower_state[2])
                entity_chosen = big_net.lower_net.sample_action(state=lower_state_tensor,
                                                                action_space=lower_action_space,
                                                                action_tensor_space=lower_action_tensor_space)[0]
                sequence_of_operands.append(entity_chosen)
                lower_next_state = (prover.raw_observe(), theorem_chosen, sequence_of_operands)
                lower_reward, reward_string = proof.apply_theorem_get_reward(theorem=theorem_chosen,
                                                                             operands=sequence_of_operands,
                                                                             reward_scheme=lower_reward_scheme)
                proof.trim_entities(max_number_of_entities=max_number_of_entities)
                proof.trim_ground_truth(max_number_of_ground_truth=max_number_of_ground_truth)
                aggregate_lower_reward += lower_reward
                aggregate_lower_steps += 1
                lower_reward = torch.FloatTensor([[lower_reward]]).to(device)
                lower_done = 1 if (operand_index == theorem_chosen.input_no) else 0
                lower_done = torch.FloatTensor([[lower_done]]).to(device)
                lower_buffer.push(state=lower_state, next_state=lower_next_state, reward=lower_reward,
                                  action=entity_chosen, done=lower_done)
                imitation_reward_tensor = torch.FloatTensor([[imitation_reward]]).to(device)
                # TODO: should only include steps on logic chain
                if lower_reward > 0:
                    lower_episode_list.append(
                        {"state": lower_state, "next_state": lower_next_state,
                         "reward": imitation_reward_tensor, "action": entity_chosen, "done": lower_done}
                    )
                single_proof_steps[-1].append(entity_chosen.name)

            # Higher MDP and update
            higher_next_state = prover.raw_observe()
            higher_done = 1 if proof.proved else 0
            higher_done = torch.FloatTensor([[higher_done]]).to(device)
            higher_reward = higher_reward_scheme[reward_string]
            aggregate_higher_reward += higher_reward
            aggregate_higher_steps += 1
            higher_reward = torch.FloatTensor([[higher_reward]]).to(device)
            higher_buffer.push(state=higher_state, next_state=higher_next_state, reward=higher_reward,
                               action=theorem_chosen, done=higher_done)
            imitation_reward_tensor = torch.FloatTensor([[imitation_reward]]).to(device)
            # TODO: should only include steps on logic chain
            if higher_reward > 0:
                higher_episode_list.append(
                    {"state": higher_state, "next_state": higher_next_state,
                     "reward": imitation_reward_tensor, "action": theorem_chosen, "done": higher_done}
                )

            # Trigger training when a theorem is proved for the first time
            if proof.proved:
                training_trigger = True
                lower_success_buffer.cache(lower_episode_list)
                higher_success_buffer.cache(higher_episode_list)
                lower_episode_list = list()
                higher_episode_list = list()
                break
        # proving_process[degree].append(proof.print_proof_status())

        training_info["lower_rewards"].append(aggregate_lower_reward / aggregate_lower_steps)
        training_info["higher_rewards"].append(aggregate_higher_reward / aggregate_higher_steps)
        training_info["optimality"].append(aggregate_higher_steps)
        training_info["proved"].append(int(proof.proved))
        training_info["updates"].append(update)

        if len(lower_buffer) >= batch_size and training_trigger:
            lower_representation_time, lower_update_time, lower_td_errors = \
                update_lower(lower_net=big_net.lower_net, buffer=lower_buffer, configuration=configuration,
                             encoder=big_net.encoder, entity_space=proof.entities,
                             optimizer=lower_encoder_optimizer, epochs=lower_epochs)
        if len(higher_buffer) >= batch_size and training_trigger:
            higher_representation_time, higher_update_time, higher_td_errors = \
                update_higher(higher_net=big_net.higher_net, buffer=higher_buffer, configuration=configuration,
                              encoder=big_net.encoder, optimizer=higher_optimizer, theorem_space=proof.lemmas,
                              epochs=higher_epochs)

        update += higher_epochs * \
                  (int(len(higher_buffer) / batch_size) + int(len(higher_success_buffer) / batch_size))
        print(update)

        if len(lower_buffer) >= batch_size and len(higher_buffer) >= batch_size and \
                len(lower_success_buffer) >= 0 and len(higher_buffer) >= batch_size and training_trigger:
            training_info["time"].append({
                "lower_rep_time": lower_representation_time / (lower_epochs * int(len(lower_buffer) / batch_size)),
                "lower_update_time": lower_update_time / (lower_epochs * int(len(lower_buffer) / batch_size)),
                "higher_rep_time": higher_representation_time / (
                        higher_epochs * int(len(higher_buffer) / batch_size)),
                "higher_update_time": higher_update_time / (higher_epochs * int(len(higher_buffer) / batch_size))
            })
            # training_info[degree]["buffer size"].append((len(lower_buffer), len(higher_buffer),
            #                                              len(lower_success_buffer), len(higher_success_buffer)))
            training_info["lower td error"].extend(lower_td_errors)
            training_info["higher td error"].extend(higher_td_errors)

        model = {
            "big_net": big_net,
            "higher_q_optimizer": higher_optimizer, "lower_encoder_q_optimizer": lower_encoder_optimizer,
            "training_info": training_info,
            "lower_buffer": lower_buffer, "higher_buffer": higher_buffer,
            "lower_success_buffer": lower_success_buffer, "higher_success_buffer": higher_success_buffer
        }
        torch.save(model, saving_directory + "model.pt")
        json.dump(training_info, open(saving_directory + "training_info{}.json".format(seed), "w"))
        total_proof_steps.append(single_proof_steps)
        json.dump(total_proof_steps, open(saving_directory + "proof_steps{}.json".format(seed), "w"))
        # json.dump(proving_process, open(saving_directory + "proving_process{}.json".format(seed), "w"))
        json.dump(proof_info, open(saving_directory + "proof_info{}.json".format(seed), "w"))


if __name__ == "__main__":
    config = load_config()["imitation_aided_cts"]
    run(config)
