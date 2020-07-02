import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))
from legacy.helper_functions.utils import set_random_seed

set_random_seed(1)


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
            action_tensor = self.encoder.encode_theorem(action).repeat(self.batch_size, 1)
            q_values.append(self.forward(state, action_tensor))
        return torch.cat(q_values, dim=1)

    def partition_value(self, state, action_space):
        exp_q_values = list()
        for action in action_space:
            action_tensor = self.encoder.encode_theorem(action).repeat(self.batch_size, 1)
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
        self.message_seq = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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
        message = self.message_seq(x)

        return energy, temp, message

    def evaluate(self, state, action_space):
        action_index_list = list(range(len(action_space)))
        energy, temp, _ = self.forward(state)
        tempered_energy = energy * temp
        prob = F.softmax(tempered_energy, dim=1).squeeze()

        if len(prob.size()) == 1:
            prob_list = prob.cpu().detach().numpy().tolist()
            if len(action_index_list) != len(prob_list):
                print(prob_list)
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
    new_action_tensor = torch.cat([encoder.encode_theorem(theorem) for theorem in new_action], dim=0)
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
