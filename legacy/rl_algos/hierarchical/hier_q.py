import json
import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

from legacy.helper_functions.QNetwork import GeneralQNetwork
from legacy.helper_functions.utils import *


def update(q_net, optimizer, action_space, action_tensor_space, buffer, configuration):
    batch_size = configuration["batch_size"]
    discount_factor = configuration["discount_factor"]
    list_of_batches = prepare_batches(buffer, batch_size)
    for batch in list_of_batches:
        dict_batch = convert_batch_dict_to_dict_batch(batch)
        state = dict_batch["state"]
        action = dict_batch["action"]
        next_state = dict_batch["next_state"]
        reward = dict_batch["reward"]
        done = dict_batch["done"]
        loss = q_net.q_loss(state=state, action=action, reward=reward, next_state=next_state, done=done,
                            action_space=action_space, action_tensor_space=action_tensor_space,
                            discount_factor=discount_factor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train(random_seed):
    set_random_seed(random_seed)
    # Loading stuff
    config = load_config()["hier_q"]
    higher_lr = config["higher_lr"]
    lower_lr = config["lower_lr"]
    capacity = config["replay_buffer_capacity"]
    saving_dir = config["saving_dir"]
    saving_affix = config["saving_affix"]
    max_frames = config["max_frames"]
    max_steps = config["max_steps"]
    lower_reward_scheme = config["lower_reward_dict"]
    higher_reward_scheme = config["higher_reward_dict"]
    batch_size = config["batch_size"]

    encoder = torch.load(config["encoder_path"])
    device = encoder.device
    for encoder_para in encoder.parameters():
        encoder_para.requires_grad = False
    original_prover = encoder.prover
    original_proof = original_prover.proof
    lower_buffer = ReplayBuffer("L1 Buffer", capacity)
    higher_buffer = ReplayBuffer("L2 Buffer", capacity)
    lower_action_space = original_proof.entities
    higher_action_space = original_proof.lemmas
    lower_action_tensor_space = torch.cat([encoder.encode_entity(entity) for entity in original_proof.entities], dim=0)
    higher_action_tensor_space = torch.cat([encoder.encode_theorem(lemma) for lemma in original_proof.lemmas], dim=0)

    # Figuring out dimensions
    state_dim = encoder.observation_tensor_size
    higher_action_dim = encoder.theorem_embedding_size
    lower_action_dim = encoder.entity_embedding_size

    # Initializing nets and optimizers
    lower_q_net = GeneralQNetwork(
        state_dim=state_dim + higher_action_dim + encoder.theorem_operand_maxsize * lower_action_dim,
        action_dim=lower_action_dim
    )
    lower_q_optimizer = torch.optim.Adam(lower_q_net.parameters(), lr=lower_lr, amsgrad=True)
    higher_q_net = GeneralQNetwork(state_dim=state_dim, action_dim=higher_action_dim)
    higher_q_optimizer = torch.optim.Adam(higher_q_net.parameters(), lr=higher_lr, amsgrad=True)

    if os.path.isfile("{}checkpoint{}.pt".format(saving_dir, saving_affix)):
        model = torch.load("{}checkpoint{}.pt".format(saving_dir, saving_affix))
        frame = model["frame"]
        higher_q_net = model["higher_q_net"]
        higher_q_optimizer = model["higher_q_optimizer"]
        lower_q_net = model["lower_q_net"]
        lower_q_optimizer = model["lower_q_optimizer"]
        lower_rewards = model["lower_rewards"]
        higher_rewards = model["higher_rewards"]
        lower_success_rate = model["lower_success_rate"]
    else:
        lower_rewards = list()
        higher_rewards = list()
        lower_success_rate = list()
        frame = 0

    while frame < max_frames:
        print(frame)
        aggregate_lower_reward = 0
        aggregate_lower_step = 0
        aggregate_higher_reward = 0
        aggregate_higher_step = 0
        lower_success_trials = 0
        lower_total_trials = 0
        prover = deepcopy(original_prover)
        proof = prover.proof
        for step in range(max_steps):
            higher_state = encoder.encode_observation(prover.raw_observe())
            lemma_chosen = higher_q_net.sample_action(state=higher_state, action_space=higher_action_space,
                                                      action_tensor_space=higher_action_tensor_space)[0]
            lemma_tensor = encoder.encode_theorem(lemma_chosen)
            sequence_of_operands = list()
            for operand_index in range(1, 1 + lemma_chosen.input_no):
                # Lower MDP and update
                sequence_tensor = encoder.encode_entities(sequence_of_operands)
                lower_state = torch.cat([higher_state, lemma_tensor, sequence_tensor], dim=-1)
                entity_chosen = lower_q_net.sample_action(state=lower_state, action_space=lower_action_space,
                                                          action_tensor_space=lower_action_tensor_space)[0]
                entity_chosen_tensor = encoder.encode_entity(entity_chosen)
                sequence_of_operands.append(entity_chosen)
                sequence_tensor = encoder.encode_entities(sequence_of_operands)
                lower_next_state = torch.cat([higher_state, lemma_tensor, sequence_tensor], dim=-1)
                lower_reward, reward_string = proof.apply_theorem_get_reward(theorem=lemma_chosen,
                                                                             operands=sequence_of_operands,
                                                                             reward_scheme=lower_reward_scheme)
                lower_reward = torch.FloatTensor([[lower_reward]]).to(device)
                aggregate_lower_reward += lower_reward
                aggregate_lower_step += 1
                lower_done = 1 if (operand_index == lemma_chosen.input_no) else 0
                lower_done = torch.FloatTensor([[lower_done]]).to(device)
                lower_buffer.push(state=lower_state, next_state=lower_next_state, reward=lower_reward,
                                  action=entity_chosen_tensor, done=lower_done)
                if len(lower_buffer) >= batch_size:
                    update(q_net=lower_q_net, optimizer=lower_q_optimizer, action_space=lower_action_space,
                           action_tensor_space=lower_action_tensor_space,
                           buffer=lower_buffer, configuration=config)
                frame += 1
            if lower_reward > 0:
                lower_success_trials += 1
            lower_total_trials += 1

            # Higher MDP and update
            higher_next_state = encoder.encode_observation(prover.raw_observe())
            higher_done = 1 if proof.proved else 0
            higher_reward = higher_reward_scheme[reward_string]
            higher_done = torch.FloatTensor([[higher_done]]).to(device)
            higher_reward = torch.FloatTensor([[higher_reward]]).to(device)
            aggregate_higher_reward += higher_reward
            aggregate_higher_step += 1
            higher_buffer.push(state=higher_state, next_state=higher_next_state, reward=higher_reward,
                               action=lemma_tensor, done=higher_done)
            if len(higher_buffer) >= batch_size:
                update(q_net=higher_q_net, optimizer=higher_q_optimizer, action_space=higher_action_space,
                       action_tensor_space=higher_action_tensor_space, buffer=higher_buffer, configuration=config)
            if proof.proved:
                break
        lower_rewards.append((aggregate_lower_reward / aggregate_lower_step).item())
        higher_rewards.append((aggregate_higher_reward / aggregate_higher_step).item())
        lower_success_rate.append(lower_success_trials / lower_total_trials)
        json.dump(lower_rewards,
                  open("{}lower_rewards{}_seed{}.json".format(saving_dir, saving_affix, random_seed), "w"))
        json.dump(higher_rewards,
                  open("{}higher_rewards{}_seed{}.json".format(saving_dir, saving_affix, random_seed), "w"))
        json.dump(lower_success_rate,
                  open("{}lower_success_rate{}_seed{}.json".format(saving_dir, saving_affix, random_seed), "w"))

        model = {
            "frame": frame, "higher_q_net": higher_q_net, "higher_q_optimizer": higher_q_optimizer,
            "lower_q_net": lower_q_net, "lower_q_optimizer": lower_q_optimizer,
            "lower_rewards": lower_rewards, "higher_rewards": higher_rewards, "lower_success_rate": lower_success_rate
        }
        torch.save(model, "{}checkpoint{}_seed{}.pt".format(saving_dir, saving_affix, random_seed))


if __name__ == "__main__":
    total_random_seeds = 1
    for seed in range(total_random_seeds):
        train(seed)
