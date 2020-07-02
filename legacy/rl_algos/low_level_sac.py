from legacy import logic_math
import random
import json
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))
from legacy.helper_functions.utils import set_random_seed, load_config, ReplayBuffer
from legacy.pseq.errors import InputError

random_seeds = 10


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.hidden_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.final_layer = nn.Linear(hidden_dim, 1)

        self.final_layer.weight.data.uniform_(-init_w, init_w)
        self.final_layer.bias.data.uniform_(-init_w, init_w)

        # Device to operate on
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = self.hidden_layers(state)
        x = self.final_layer(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, batch_size, encoder, state_dim, action_dim, hidden_dim, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.batch_size = batch_size
        self.encoder = encoder

        self.hidden_layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.final_layer = nn.Linear(hidden_dim, 1)
        self.final_layer.weight.data.uniform_(-init_w, init_w)
        self.final_layer.bias.data.uniform_(-init_w, init_w)

        # Device to operate on
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.hidden_layers(x)
        x = self.final_layer(x)
        return x

    def q_value_vector(self, state, action_space):
        q_values = list()
        for action in action_space:
            action_tensor = self.encoder.encode_entity(action).repeat(self.batch_size, 1)
            q_values.append(self.forward(state, action_tensor))
        return torch.cat(q_values, dim=1)

    def partition_value(self, state, action_space):
        exp_q_values = list()
        for action in action_space:
            action_tensor = self.encoder.encode_entity(action).repeat(self.batch_size, 1)
            exp_q_values.append(self.forward(state, action_tensor).exp())
        exp_concat = torch.cat(exp_q_values, dim=1)
        return torch.sum(input=exp_concat, dim=1)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim, energy_min=-10., energy_max=10.,
                 temp_min=1e-3, temp_max=1.):
        super(PolicyNetwork, self).__init__()

        self.energy_min = energy_min
        self.energy_max = energy_max
        self.temp_min = temp_min
        self.temp_max = temp_max

        self.hidden_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.energy_linear = nn.Linear(hidden_dim, num_actions)
        self.temp_linear = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Device to operate on
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = self.hidden_layers(state)
        energy = self.energy_linear(x)
        energy = torch.clamp(energy, self.energy_min, self.energy_max)
        temp = self.temp_linear(x)
        temp = temp * self.temp_max

        return energy, temp

    def evaluate(self, state, action_space):
        action_index_list = list(range(len(action_space)))
        energy, temp = self.forward(state)
        tempered_energy = energy * temp
        prob = F.softmax(tempered_energy, dim=1).squeeze()

        if len(prob.size()) == 1:
            prob_list = prob.cpu().detach().numpy().tolist()
            new_action_index = random.choices(action_index_list, prob_list, k=1)[0]
            new_action = action_space[new_action_index]
            new_action_prob = prob[new_action_index]
            return new_action, new_action_prob, prob
        else:
            new_actions = list()
            new_action_probs = list()
            prob_list = prob.cpu().detach().numpy().tolist()
            for dim0_index in range(prob.size(0)):
                new_action_index = random.choices(action_index_list, prob_list[dim0_index], k=1)[0]
                new_action = action_space[new_action_index]
                new_action_prob = prob[dim0_index][new_action_index].unsqueeze(-1).unsqueeze(-1)
                new_actions.append(new_action)
                new_action_probs.append(new_action_prob)
            new_action_probs = torch.cat(new_action_probs, dim=0)
            return new_actions, new_action_probs, prob

    def get_action(self, state, action_space):
        new_action, _, _ = self.evaluate(state, action_space)
        return new_action


def soft_q_update(encoder, replay_buffer, batch_size, device,
                  soft_q_net, value_net, target_value_net, policy_net,
                  soft_q_optimizer, value_optimizer, policy_optimizer,
                  soft_q_criterion, value_criterion,
                  action_space,
                  gamma=0.99,
                  soft_tau=1e-2):
    batch_dict = replay_buffer.sample(batch_size)
    state_tensor, action_tensor, reward, next_state_tensor, done = batch_dict["state"], batch_dict["action"], \
                                                                   batch_dict["reward"], batch_dict["next_state"], \
                                                                   batch_dict["done"]

    state_tensor = torch.cat(state_tensor, dim=0).to(device)
    next_state_tensor = torch.cat(next_state_tensor, dim=0).to(device)
    action_tensor = torch.cat(action_tensor, dim=0).to(device)
    reward = torch.FloatTensor(np.float32(reward)).unsqueeze(1).to(device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    # Update soft q network
    expected_q_value = soft_q_net(state_tensor, action_tensor)
    target_value = target_value_net(next_state_tensor)
    next_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss = soft_q_criterion(expected_q_value, next_q_value.detach())
    soft_q_optimizer.zero_grad()
    q_value_loss.backward()
    soft_q_optimizer.step()

    # Update value network
    expected_value = value_net(state_tensor)
    new_action, new_action_prob, prob = policy_net.evaluate(state_tensor, action_space)
    print(prob[0])
    print(new_action[0].name, "\n")
    new_action_log_prob, log_prob = torch.log(new_action_prob), torch.log(prob)
    new_action_tensor = torch.cat([encoder.encode_entity(entity) for entity in new_action], dim=0)
    expected_new_q_value = soft_q_net(state_tensor, new_action_tensor)
    next_value = expected_new_q_value - new_action_log_prob
    value_loss = value_criterion(expected_value, next_value.detach())
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    # Update policy network
    log_partition_value = torch.log(soft_q_net.partition_value(state_tensor, action_space))
    q_value_vector = soft_q_net.q_value_vector(state_tensor, action_space)
    policy_loss = (log_partition_value * (torch.sum(prob * (log_prob - q_value_vector), dim=-1))).mean()
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    # Polyak average target network
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


def train(random_seed=1):
    set_random_seed(random_seed)
    random_seed = "random_seed_{}".format(random_seed)
    config = load_config()["soft_actor_critic"]
    hidden_dim = config["hidden_dim"]
    value_lr = config["value_lr"]
    soft_q_lr = config["soft_q_lr"]
    policy_lr = config["policy_lr"]
    replay_buffer_capacity = config["replay_buffer_capacity"]
    max_frames = config["max_frames"]
    max_steps = config["max_steps"]
    batch_size = config["batch_size"]
    saving_dir = config["saving_dir"]
    saving_affix = config["saving_affix"]
    reward_dict = config["reward_dict"]

    # Load encoder and disable its training
    encoder = torch.load(config["encoder_path"])
    for encoder_para in encoder.parameters():
        encoder_para.requires_grad = False

    # Device to operate on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = encoder.theorem_embedding_size + encoder.theorem_operand_maxsize * encoder.entity_embedding_size
    action_dim = encoder.entity_embedding_size
    num_actions = len(encoder.theorems)
    value_criterion = nn.MSELoss()
    soft_q_criterion = nn.MSELoss()

    # Define networks
    value_net = ValueNetwork(state_dim, hidden_dim)
    target_value_net = ValueNetwork(state_dim, hidden_dim)
    soft_q_net = SoftQNetwork(batch_size, encoder, state_dim, action_dim, hidden_dim)
    policy_net = PolicyNetwork(state_dim, len(encoder.prover.proof.entities), hidden_dim)

    # Initialize target network to have the same parameters with the value network
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(param.data)

    value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
    soft_q_optimizer = optim.Adam(soft_q_net.parameters(), lr=soft_q_lr)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(name="low level buffer", capacity=replay_buffer_capacity)

    # Assume checkpointed state
    if os.path.isfile("{}checkpoint{}{}.pt".format(saving_dir, saving_affix, random_seed)):
        state = torch.load("{}checkpoint{}{}.pt".format(saving_dir, saving_affix, random_seed))
        value_net = state["value_net"]
        target_value_net = state["target_value_net"]
        soft_q_net = state["soft_q_net"]
        policy_net = state["policy_net"]
        value_optimizer = state["value_optimizer"]
        soft_q_optimizer = state["soft_q_optimizer"]
        policy_optimizer = state["policy_optimizer"]
        frame_idx = state["frame_idx"]
        rewards = state["rewards"]
        success_rate = state["success_rate"]
    else:
        frame_idx = 0
        rewards = list()
        success_rate = list()

    while frame_idx < max_frames:
        prover_copy = deepcopy(encoder.prover)
        proof = prover_copy.proof
        episode_reward = 0
        success_proceedings = 0
        total_proceedings = 0

        for step in range(max_steps):
            theorem = random.choices(population=proof.lemmas, k=1)[0]
            entities_taken = list()
            for entity_idx in range(1, 1 + theorem.input_no):
                state_tensor = torch.cat([encoder.encode_theorem(theorem), encoder.encode_entities(entities_taken)],
                                         dim=-1)
                action = policy_net.get_action(state=state_tensor, action_space=proof.entities)
                action_tensor = encoder.encode_entity(action)
                entities_taken += [action]
                next_state_tensor = torch.cat([encoder.encode_theorem(theorem), encoder.encode_entities(entities_taken)]
                                              , dim=-1)

                # Get reward for the transition
                if entity_idx == theorem.input_no:
                    done = True
                    try:
                        reward_string = prover_copy.proof.apply_theorem(theorem=theorem, operands=entities_taken)
                    except InputError:
                        reward_string = "REWARD_INPUT_INVALID"
                else:
                    done = False
                    reward_string = "REWARD_NULL"
                reward = reward_dict[reward_string]
                if reward > 0:
                    reward = reward * logic_math.exp(theorem.input_no)
                else:
                    pass

                # Add transition to the replay buffer
                replay_buffer.push(state=state_tensor, action=action_tensor, next_state=next_state_tensor,
                                   reward=reward, done=done)
                if len(replay_buffer) > batch_size:
                    soft_q_update(encoder=encoder, replay_buffer=replay_buffer, batch_size=batch_size, device=device,
                                  soft_q_net=soft_q_net, value_net=value_net,
                                  target_value_net=target_value_net, policy_net=policy_net,
                                  soft_q_optimizer=soft_q_optimizer, value_optimizer=value_optimizer,
                                  policy_optimizer=policy_optimizer,
                                  soft_q_criterion=soft_q_criterion, value_criterion=value_criterion,
                                  action_space=proof.entities)
                frame_idx += 1
            episode_reward += reward
            if reward > 0:
                success_proceedings += 1
            elif reward < 0:
                pass
            else:
                raise InputError
            total_proceedings += 1

        rewards.append(episode_reward)
        success_rate.append(success_proceedings / total_proceedings)

        state = {
            "frame_idx": frame_idx, "rewards": rewards, "success_rate": success_rate,
            "value_net": value_net, "target_value_net": target_value_net,
            "soft_q_net": soft_q_net, "policy_net": policy_net,
            "value_optimizer": value_optimizer, "soft_q_optimizer": soft_q_optimizer,
            "policy_optimizer": policy_optimizer
        }
        torch.save(state, "{}checkpoint{}{}.pt".format(saving_dir, saving_affix, random_seed))
        json.dump(rewards, open("{}rewards{}{}".format(saving_dir, saving_affix, random_seed), "w"))
        json.dump(success_rate, open("{}success_rate{}{}".format(saving_dir, saving_affix, random_seed), "w"))
    torch.save(state, "{}model{}{}.pt".format(saving_dir, saving_affix, random_seed))


if __name__ == "__main__":
    for i in range(1, 1 + random_seeds):
        train(i)
