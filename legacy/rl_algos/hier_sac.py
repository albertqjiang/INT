from legacy import logic_math
import json
import torch
import torch.optim as optim
import torch.nn as nn

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))
from legacy.helper_functions.utils import load_config, ReplayBuffer
import legacy.rl_algos.low_level_sac as l_sac
import legacy.rl_algos.high_level_sac as h_sac
from legacy.pseq.errors import InputError


def train():
    scheme = "hier_sac_pretrained"
    pretrained = scheme.endswith("pretrained")
    config = load_config()[scheme]
    hidden_dim = config["hidden_dim"]
    value_lr = config["value_lr"]
    soft_q_lr = config["soft_q_lr"]
    policy_lr = config["policy_lr"]
    replay_buffer_capacity = config["replay_buffer_capacity"]
    batch_size = config["batch_size"]
    saving_dir = config["saving_dir"]
    saving_affix = config["saving_affix"]
    lower_reward_dict = config["lower_reward_dict"]
    higher_reward_dict = config["higher_reward_dict"]
    lower_rewards_scale = config["lower_rewards_scale"]
    higher_rewards_scale = config["higher_rewards_scale"]
    entropy_scaling = config["entropy_scaling"]

    # Load encoder and disable its training
    # This is only for pre-trained rl, comment it out if training from scratch
    encoder = torch.load(config["encoder_path"])
    for encoder_para in encoder.parameters():
        encoder_para.requires_grad = False

    prover = encoder.prover

    # Device to operate on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define dimensions
    lower_state_dim = encoder.theorem_embedding_size + \
                      encoder.theorem_operand_maxsize * encoder.entity_embedding_size
    lower_action_dim = encoder.entity_embedding_size
    higher_state_dim = encoder.entity_embedding_size * encoder.ground_truth_maxsize + \
                       encoder.theorem_embedding_size * encoder.theorem_maxsize + \
                       encoder.entity_embedding_size * encoder.entity_maxsize + \
                       encoder.entity_embedding_size * encoder.objective_maxsize
    higher_action_dim = encoder.theorem_embedding_size
    value_criterion = nn.MSELoss()
    soft_q_criterion = nn.MSELoss()

    # Define networks
    if pretrained:
        pretrained_model = torch.load(config["lower_sac_model"])
        lower_value_net = pretrained_model["value_net"]
        lower_target_value_net = pretrained_model["target_value_net"]
        lower_soft_q_net = pretrained_model["soft_q_net"]
        lower_policy_net = pretrained_model["policy_net"]
        lower_value_optimizer = pretrained_model["value_optimizer"]
        lower_soft_q_optimizer = pretrained_model["soft_q_optimizer"]
        lower_policy_optimizer = pretrained_model["policy_optimizer"]
    else:
        lower_value_net = l_sac.ValueNetwork(lower_state_dim, hidden_dim)
        lower_target_value_net = l_sac.ValueNetwork(lower_state_dim, hidden_dim)
        lower_soft_q_net = l_sac.SoftQNetwork(batch_size, encoder, lower_state_dim, lower_action_dim, hidden_dim)
        lower_policy_net = l_sac.PolicyNetwork(lower_state_dim, len(encoder.prover.proof.entities), hidden_dim)
        lower_value_optimizer = optim.Adam(lower_value_net.parameters(), lr=value_lr)
        lower_soft_q_optimizer = optim.Adam(lower_soft_q_net.parameters(), lr=soft_q_lr)
        lower_policy_optimizer = optim.Adam(lower_policy_net.parameters(), lr=policy_lr)

    higher_value_net = h_sac.ValueNetwork(higher_state_dim, hidden_dim)
    higher_target_value_net = h_sac.ValueNetwork(higher_state_dim, hidden_dim)
    higher_soft_q_net = h_sac.SoftQNetwork(batch_size, encoder, higher_state_dim, higher_action_dim, hidden_dim)
    higher_policy_net = h_sac.PolicyNetwork(higher_state_dim, len(encoder.prover.proof.lemmas), hidden_dim)

    # Initialize target network to have the same parameters with the value network
    for target_param, param in zip(lower_target_value_net.parameters(), lower_value_net.parameters()):
        target_param.data.copy_(param.data)
    for target_param, param in zip(higher_target_value_net.parameters(), higher_value_net.parameters()):
        target_param.data.copy_(param.data)

    higher_value_optimizer = optim.Adam(higher_value_net.parameters(), lr=value_lr)
    higher_soft_q_optimizer = optim.Adam(higher_soft_q_net.parameters(), lr=soft_q_lr)
    higher_policy_optimizer = optim.Adam(higher_policy_net.parameters(), lr=policy_lr)

    # Initialize replay buffer
    lower_replay_buffer = ReplayBuffer(name="low level buffer", capacity=replay_buffer_capacity)
    higher_replay_buffer = ReplayBuffer(name="high level buffer", capacity=replay_buffer_capacity)

    # Assume checkpointed state
    if os.path.isfile("{}checkpoint{}.pt".format(saving_dir, saving_affix)):
        state = torch.load("{}checkpoint{}.pt".format(saving_dir, saving_affix))
        trial = state["trial"]
        lower_value_net = state["lower_value_net"]
        lower_target_value_net = state["lower_target_value_net"]
        lower_soft_q_net = state["lower_soft_q_net"]
        lower_policy_net = state["lower_policy_net"]
        lower_value_optimizer = state["lower_value_optimizer"]
        lower_soft_q_optimizer = state["lower_soft_q_optimizer"]
        lower_policy_optimizer = state["lower_policy_optimizer"]
        higher_value_net = state["higher_value_net"]
        higher_target_value_net = state["higher_target_value_net"]
        higher_soft_q_net = state["higher_soft_q_net"]
        higher_policy_net = state["higher_policy_net"]
        higher_value_optimizer = state["higher_value_optimizer"]
        higher_soft_q_optimizer = state["higher_soft_q_optimizer"]
        higher_policy_optimizer = state["higher_policy_optimizer"]
        lower_rewards = state["lower_rewards"]
        higher_rewards = state["higher_rewards"]
        success_rate = state["success_rate"]
        episode_rewards = state["episode_rewards"]
    else:
        trial = 0
        episode_rewards = list()
        lower_rewards = list()
        higher_rewards = list()
        success_rate = list()

    while trial < 10:
        episode_reward = 0
        episode_step = 0
        success_proceedings = 0
        total_proceedings = 0
        trial += 1
        while not prover.get_status():
            total_proceedings += 1
            episode_step += 1
            higher_observation = prover.raw_observe()
            higher_state_tensor = encoder.encode_observation(higher_observation).view(1, -1)
            theorem = higher_policy_net.get_action(state=higher_state_tensor, action_space=prover.proof.lemmas)
            entities_taken = list()
            for entity_idx in range(1, 1 + theorem.input_no):
                lower_state_tensor = torch.cat(
                    [encoder.encode_theorem(theorem), encoder.encode_entities(entities_taken)], dim=-1)
                lower_action = lower_policy_net.get_action(state=lower_state_tensor, action_space=prover.proof.entities)
                lower_action_tensor = encoder.encode_entity(lower_action)
                entities_taken += [lower_action]
                lower_next_state_tensor = torch.cat(
                    [encoder.encode_theorem(theorem), encoder.encode_entities(entities_taken)], dim=-1)

                # Get reward for the lower transition
                if entity_idx == theorem.input_no:
                    lower_done = True
                    try:
                        reward_string = prover.proof.apply_theorem(theorem=theorem, operands=entities_taken)
                    except InputError:
                        reward_string = "REWARD_INPUT_INVALID"
                else:
                    lower_done = False
                    reward_string = "REWARD_NULL"
                lower_reward = lower_reward_dict[reward_string] * lower_rewards_scale
                if lower_reward > 0:
                    success_proceedings += 1
                    lower_reward = lower_reward * logic_math.exp(theorem.input_no)
                else:
                    pass

                if not pretrained:
                    # Add transition to the lower replay buffer and update nets
                    lower_replay_buffer.push(state=lower_state_tensor, action=lower_action_tensor,
                                             next_state=lower_next_state_tensor, reward=lower_reward, done=lower_done)
                    if len(lower_replay_buffer) > batch_size:
                        l_sac.soft_q_update(encoder=encoder, replay_buffer=lower_replay_buffer, batch_size=batch_size,
                                            device=device, soft_q_net=lower_soft_q_net, value_net=lower_value_net,
                                            target_value_net=lower_target_value_net, policy_net=lower_policy_net,
                                            soft_q_optimizer=lower_soft_q_optimizer,
                                            value_optimizer=lower_value_optimizer,
                                            policy_optimizer=lower_policy_optimizer,
                                            soft_q_criterion=soft_q_criterion, value_criterion=value_criterion,
                                            action_space=prover.proof.entities)
            print(reward_string)
            lower_rewards.append(lower_reward)

            # Add transition to the higher replay buffer and update nets
            higher_next_observation = prover.raw_observe()
            higher_next_state_tensor = encoder.encode_observation(higher_next_observation).view(1, -1)
            higher_action_tensor = encoder.encode_theorem(theorem)
            higher_reward = higher_reward_dict[reward_string] * higher_rewards_scale
            episode_reward += higher_reward
            higher_rewards.append(higher_reward)
            higher_done = prover.get_status()
            higher_replay_buffer.push(state=higher_state_tensor, action=higher_action_tensor,
                                      next_state=higher_next_state_tensor, reward=higher_reward, done=higher_done)
            if len(higher_replay_buffer) > batch_size:
                h_sac.soft_q_update(encoder=encoder, replay_buffer=higher_replay_buffer, batch_size=batch_size,
                                    device=device, soft_q_net=higher_soft_q_net, value_net=higher_value_net,
                                    target_value_net=higher_target_value_net, policy_net=higher_policy_net,
                                    soft_q_optimizer=higher_soft_q_optimizer, value_optimizer=higher_value_optimizer,
                                    policy_optimizer=higher_policy_optimizer,
                                    soft_q_criterion=soft_q_criterion, value_criterion=value_criterion,
                                    action_space=prover.proof.lemmas)
        success_rate.append(success_proceedings / total_proceedings)
        episode_reward = episode_reward / episode_step
        episode_rewards.append(episode_reward)
        state = {
            "trial": trial, "lower_value_net": lower_value_net, "lower_target_value_net": lower_target_value_net,
            "lower_soft_q_net": lower_soft_q_net, "lower_policy_net": lower_policy_net,
            "lower_value_optimizer": lower_value_optimizer, "lower_soft_q_optimizer": lower_soft_q_optimizer,
            "lower_policy_optimizer": lower_policy_optimizer,
            "higher_value_net": higher_value_net, "higher_target_value_net": higher_target_value_net,
            "higher_soft_q_net": higher_soft_q_net, "higher_policy_net": higher_policy_net,
            "higher_value_optimizer": higher_value_optimizer, "higher_soft_q_optimizer": higher_soft_q_optimizer,
            "higher_policy_optimizer": higher_policy_optimizer,
            "episode_rewards": episode_rewards, "lower_rewards": lower_rewards, "higher_rewards": higher_rewards
        }
        torch.save(state, "{}checkpoint{}.pt".format(saving_dir, saving_affix))
        json.dump(lower_rewards, open("{}lower_rewards{}".format(saving_dir, saving_affix), "w"))
        json.dump(higher_rewards, open("{}higher_rewards{}".format(saving_dir, saving_affix), "w"))
        json.dump(episode_rewards, open("{}episode_rewards{}".format(saving_dir, saving_affix), "w"))
        json.dump(success_rate, open("{}success_rate{}".format(saving_dir, saving_affix), "w"))
        prover.reset()


if __name__ == "__main__":
    train()
